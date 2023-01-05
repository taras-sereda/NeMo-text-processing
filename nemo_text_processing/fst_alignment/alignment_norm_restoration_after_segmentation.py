import json
import os
import pickle
import re
import string
from argparse import ArgumentParser
from time import perf_counter
from typing import List, Tuple

import pynini
from alignment import create_symbol_table, get_string_alignment, get_word_segments, indexed_map_to_output, remove
from joblib import Parallel, delayed
from nemo_text_processing.text_normalization.normalize import Normalizer
from tqdm import tqdm

NA = "n/a"

"""
python alignment_norm_restoration_after_segmentation.py \
    --raw_m=/media/ebakhturina/DATA/mlops_data/pc_retained/raw.json \
    --segmented_m=/media/ebakhturina/DATA/mlops_data/pc_retained/segmented.json \
    --output_m=output.json
"""


def parse_args():
    parser = ArgumentParser("Restoration of written form after segmentation")
    parser.add_argument(
        "--raw_m",
        type=str,
        required=True,
        help=".json manifest with long audio files, required 'text' and 'audio_filepath' fields",
    )
    parser.add_argument(
        "--segmented_m",
        type=str,
        required=True,
        help=".json manifest with segmented audio files, required 'text' and 'audio_filepath' fields",
    )
    parser.add_argument(
        '--output_m', help="Path to a .json manifest to save restored output", type=str, required=True,
    )
    parser.add_argument("--language", help="language", choices=["en", "de", "es", "zh"], default="en", type=str)
    parser.add_argument("--verbose", help="print info for debugging", action='store_true')
    parser.add_argument(
        "--whitelist",
        help="path to a file with with whitelist",
        default="../text_normalization/en/data/whitelist/asr_with_pc.tsv",
        type=str,
    )
    parser.add_argument(
        "--cache_dir",
        help="path to a dir with .far grammar file. Set to None to avoid using cache",
        default="cache_dir",
        type=str,
    )
    parser.add_argument("--no_cache", action="store_true", help="Set to False to disable caching for fst alignments")
    parser.add_argument("--n_jobs", default=-2, type=int, help="The maximum number of concurrently running jobs")
    parser.add_argument("--batch_size", default=200, type=int, help="Number of examples for each process")
    parser.add_argument(
        "--with_normalizer",
        action="store_true",
        help="This flag will help restore written form for cases where input is reordered during normalization, e.g. '$5' -> `five dollars` (not dollar five). This flag increasing processing time.",
    )

    return parser.parse_args()


def remove_punctuation(text: str, remove_spaces: bool = True, do_lower: bool = True, exclude: bool = None):
    """
    Remove punctuation from text
    Args:
        text: input text
        remove_spaces: set to True to remove spaces
        do_lower: set to True to lower case
        exclude: specify a list of punctuation marks to keep, e.g. exclude = ["'", "-"]
    """
    all_punct_marks = string.punctuation

    if exclude is not None:
        for p in exclude:
            all_punct_marks = all_punct_marks.replace(p, "")

        # a weird bug where commas is getting deleted when dash is present in the list of punct marks
        all_punct_marks = all_punct_marks.replace("-", "")
    text = re.sub("[" + all_punct_marks + "]", " ", text)

    text = re.sub(r" +", " ", text)
    if remove_spaces:
        text = text.replace(" ", "").replace("\u00A0", "").strip()

    if do_lower:
        text = text.lower()
    return text.strip()


def clean(text: str):
    """ This processing shouldn't change the length of the input text"""
    text = text.replace("-", " ").replace(":", " ")
    return text


def pc_clean(text: str):
    text = (
        text.replace("-", " ")
        .replace(":", " ")
        .replace(" .", ".")
        .replace(" ,", ",")
        .replace(" ?", "?")
        .replace("  ", " ")
    )
    return text


def build_output_raw_map(alignment: List[Tuple[str, str]], output_text: str, text: str) -> List[List[str]]:
    """
    For every word in the normalized text, find corresponding raw (written) word
    Args:
        alignment: alignment fst
        output_text: normalized (spoken) text
        text: raw (written) text

    Returns: spoken - written pairs
    """
    # get TN alignment
    indices = get_word_segments(text)
    output_raw_map = []

    for i, x in enumerate(indices):
        start, end = indexed_map_to_output(start=x[0], end=x[1], alignment=alignment)
        output_raw_map.append([output_text[start:end], text[x[0] : x[1]]])
    return output_raw_map


def process_with_normalizer(
    item, key, normalizer, offset=3, output_raw_map=None, segmented_field="segmented", raw_text_field="raw_text"
):
    """
    This a restoration based on re-running normalization on segments, helps to retrieve written form for segments,
    where words were reordered during original normalization, e.g. money, dates.

    :param item:
    :param key:
    :param normalizer:
    :param offset:
    :param output_raw_map:
    :param segmented_field:
    :param raw_text_field:
    :return:
    """
    restored = []
    text = item[raw_text_field]
    segmented = item[segmented_field]

    if output_raw_map is None:
        fst = pynini.compose(normalizer.tagger.fst, normalizer.verbalizer.fst)
        table = create_symbol_table()
        alignment, output_text = get_string_alignment(fst=fst, input_text=text, symbol_table=table)

        output_raw_map = build_output_raw_map(alignment, output_text, text)
        if output_raw_map is None:
            return (key, restored)

    last_found_start_idx = 0
    for segment_id in range(0, len(segmented)):
        restored_raw = NA
        segment = segmented[segment_id]
        segment_list = segment.split()

        end_found = False
        if len(segment_list) == 0:
            restored.append(restored_raw)
            continue
        first_word = segment_list[0]
        for id in [
            i for i, x in enumerate(output_raw_map[last_found_start_idx:]) if first_word.lower() in x[0].lower()
        ]:
            if end_found:
                break
            end_idx = id + (len(segment_list) - offset)
            while not end_found and end_idx <= len(output_raw_map):
                restored_norm = " ".join([x[0] for x in output_raw_map[last_found_start_idx:][id:end_idx]])
                restored_raw = " ".join([x[1] for x in output_raw_map[last_found_start_idx:][id:end_idx]])

                processed_raw = pc_clean(normalizer.normalize(restored_raw).lower())
                processed_segment = pc_clean(segment.lower())
                processed_restored = pc_clean(restored_norm.lower())

                if processed_restored == processed_segment or processed_raw == processed_segment:
                    end_found = True
                    last_found_start_idx = end_idx
                elif processed_segment.startswith(processed_restored) or processed_segment.startswith(processed_raw):
                    end_idx += 1
                else:
                    restored_raw = NA
                    break
        if end_found:
            restored.append(restored_raw)
    return (key, restored)


def get_raw_text_from_alignment(alignment, alignment_start_idx=0, alignment_end_idx=None):
    if alignment_end_idx is None:
        alignment_end_idx = len(alignment)
    return "".join(list(map(remove, [x[0] for x in alignment[alignment_start_idx : alignment_end_idx + 1]])))


def process(item, key, normalizer, use_cache=True, verbose=False, with_normalizer=True, cache_dir=""):
    start_time = perf_counter()
    restored = {}
    text = item["raw_text"]

    if "segmented" not in item:
        return (key, restored)

    segmented = item["segmented"]
    cached_alignment = f"{cache_dir}/alignment_{key}.p"

    fst = pynini.compose(normalizer.tagger.fst, normalizer.verbalizer.fst)
    if os.path.exists(cached_alignment) and use_cache:
        alignment = pickle.load(open(cached_alignment, "rb"))
    else:
        print(f"generating alignments {key}")
        table = create_symbol_table()
        alignment, _ = get_string_alignment(fst=fst, input_text=text, symbol_table=table)
        pickle.dump(alignment, open(cached_alignment, "wb"))
        if verbose:
            print(f"alignment for {key}output saved to {cached_alignment}")

    segmented_result = []
    segmented_indices = []
    for i, x in enumerate(alignment):
        value = remove(x[1])
        if value != "":
            segmented_result.append(value)
            segmented_indices.append(i)

    segmented_result = "".join(segmented_result)
    failed = []
    alignment_end_idx = None
    segmented_result_clean = clean(segmented_result).lower()
    for id, segment in enumerate(segmented):
        segment_clean = clean(segment).lower()
        if segment_clean in segmented_result_clean:
            segment_start_idx = max(segmented_result_clean.index(segment_clean) - 1, 0)

            alignment_start_idx = segmented_indices[segment_start_idx]
            segment_end_idx = min(segment_start_idx + len(segment), len(segmented_indices) - 1)
            alignment_end_idx = segmented_indices[segment_end_idx]

            raw_text = "".join(
                list(map(remove, [x[0] for x in alignment[alignment_start_idx : alignment_end_idx + 1]]))
            )
            restored[id] = raw_text.strip()
            if verbose:
                print("=" * 40)
                print("FOUND:")
                print(f"RAW : {raw_text}")
                print(f"SEGM: {segment}")
                print("=" * 40)

            if len(failed) > 0 and len(failed[-1]) == 3:
                failed[-1].append(alignment_start_idx)
                idx = len(failed) - 2
                while idx >= 0 and len(failed[idx]) == 3:
                    failed[idx].append(alignment_start_idx)
                    idx -= 1

        elif alignment_end_idx is not None:
            failed.append([id, segment, alignment_end_idx])
            if id == len(segmented) - 1:
                failed[-1].append(len(alignment))
                idx = len(failed) - 2
                while idx >= 0 and len(failed[idx]) == 3:
                    failed[idx].append(alignment_start_idx)
                    idx -= 1

    num_failed = len(failed)

    if with_normalizer:
        print(f"{key} failed before with_normalizer: {num_failed}")
        num_failed = 0
        for i in range(len(failed)):
            alignment_start_idx, alignment_end_idx = failed[i][2], failed[i][3]
            raw_text_ = get_raw_text_from_alignment(
                alignment, alignment_start_idx=alignment_start_idx, alignment_end_idx=alignment_end_idx
            )
            alignment_current = alignment[alignment_start_idx : alignment_end_idx + 1]
            output_norm_current = "".join(map(remove, [x[1] for x in alignment_current]))
            tmp_item = {"raw_text": raw_text_, "segmented": [failed[i][1]], "misc": ""}

            output_raw_map = build_output_raw_map(alignment_current, output_norm_current, raw_text_)

            if output_raw_map is None:
                continue

            failed_restored = (process_with_normalizer(tmp_item, "debug", normalizer, output_raw_map=output_raw_map))[
                1
            ]
            if len(failed_restored) > 0 and failed_restored[-1] != NA:
                restored[failed[i][0]] = failed_restored[-1].strip()
                if verbose:
                    print("=" * 40)
                    print(f"RAW : {raw_text_}")
                    print(f"SEGM: {failed[i][1]}")
                    print("=" * 40)
            else:
                num_failed += 1

    for i in range(len(segmented)):
        if i in restored:
            item["misc"][i]["text_pc"] = restored[i]

    result = [x for x in item["misc"] if "text_pc" in x]
    if verbose:
        print(
            f"{key} -- found: {len(result)} out of {len(item['segmented'])} ({round(len(result) / len(item['segmented']) * 100, 1)}%)"
        )
    print(
        f'Restoration {key}: {round((perf_counter() - start_time), 2)} sec -- {len(item["segmented"])} -- failed: {num_failed}'
    )
    return result


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.raw_m):
        raise ValueError(f"{args.raw_m} not found")

    if not os.path.exists(args.segmented_m):
        raise ValueError(f"{args.segmented_m} not found")

    # for every audio file store "raw" and "segmented" samples
    # data["Am4BKyvYBgY"].keys()
    # >> dict_keys(['raw_text', 'segmented'])

    data = {}
    # read raw manifest
    with open(args.raw_m, "r") as f_in:
        for line in f_in:
            line = json.loads(line)
            text = re.sub(r" +", " ", line["text"])
            audio = line["audio_filepath"].split("/")[-1].replace(".wav", "")
            data[audio] = {"raw_text": text}

    segmented = {}
    misc_data = {}
    # read manifest after segmentation
    with open(args.segmented_m, "r") as f_in:
        for line in f_in:
            line = json.loads(line)
            text = re.sub(r" +", " ", line["text"])
            audio = "_".join(line["audio_filepath"].split("/")[-1].split("_")[:-1])

            if audio not in segmented:
                segmented[audio] = []
                misc_data[audio] = []
            segmented[audio].append(line["text"])
            misc_data[audio].append(line)

    for audio in segmented:
        segmented_lines = segmented[audio]
        misc_line_data = misc_data[audio]

        if audio not in data:
            print(f"{audio} from {args.segmented_m} is missing in the {args.raw_m}")
        else:
            data[audio]["segmented"], data[audio]["misc"] = [], []
            for segm, misc in zip(segmented_lines, misc_line_data):
                if len(segm) > 0:
                    data[audio]["segmented"].append(segm)
                    data[audio]["misc"].append(misc)

    # remove data where there are no corresponding segmented samples
    audio_data_to_del = [audio for audio in data if "segmented" not in data[audio].keys()]
    print(f"No corresponding segments found for {audio_data_to_del}, removing")
    for key in audio_data_to_del:
        del data[key]

    # normalize raw manifest for alignment
    normalizer = Normalizer(input_case="cased", cache_dir=args.cache_dir, overwrite_cache=False, lang=args.language)

    start_time = perf_counter()
    result = Parallel(n_jobs=args.n_jobs)(
        delayed(process)(
            item,
            key,
            normalizer,
            use_cache=not args.no_cache,
            verbose=args.verbose,
            with_normalizer=args.with_normalizer,
            cache_dir=args.cache_dir,
        )
        for key, item in tqdm(data.items())
    )
    print(f'Restoration parallel: {round((perf_counter() - start_time), 2)} sec.')
    print(f"use_cache: {not args.no_cache}, verbose: {args.verbose}, with_normalizer: {args.with_normalizer}")

    # start validation
    print("Starting validation...")
    start_validation_time = perf_counter()
    restored_lines = []
    for audio_segments in result:
        for line in audio_segments:
            restored_lines.append(line["text_pc"])

    # this is needed to validate alignments
    normalizer_prediction = normalizer.normalize_list(
        restored_lines,
        verbose=False,
        punct_pre_process=False,
        punct_post_process=False,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
    )

    drop_during_validation_other_error = []
    drop_during_validation_text_error = []
    num_restored = 0
    idx = 0
    # save result
    with open(args.output_m, "w") as f_out:
        for audio_segments in result:
            for line in audio_segments:
                normalized_pc = normalizer_prediction[idx]
                if remove_punctuation(line["text"]) != remove_punctuation(normalized_pc):
                    if len(line["text"]) < len(normalized_pc) and line["text"].lower() == normalized_pc[1:].lower():
                        drop_during_validation_text_error.append((line["text"], normalized_pc))
                    else:
                        drop_during_validation_other_error.append((line["text"], normalized_pc))
                else:
                    f_out.write(json.dumps(line, ensure_ascii=False) + '\n')
                    num_restored += 1
                idx += 1

    if args.verbose:
        for d in drop_during_validation_other_error:
            print(f'TEXT: {d[0]}')
            print(f'RESTORED: {d[1]}')
            print("-" * 40)

    # samples with drop_text_error
    segments_with_error = []
    text_errors = [x[0] for x in drop_during_validation_text_error]
    for samples_per_audio in misc_data.values():
        for sample in samples_per_audio:
            for text_, restored_ in drop_during_validation_text_error:
                if text_ == sample["text"]:
                    segments_with_error.append((sample, restored_))
                    break

    num_segmented_original = sum([len(item['segmented']) for item in data.values()])
    print(
        f"Validation: dropped_other_error {len(drop_during_validation_other_error)},"
        f"dropped_text_error: {len(drop_during_validation_text_error)},"
        f"algo drop: {num_segmented_original-len(drop_during_validation_other_error)-len(drop_during_validation_text_error)}"
    )

    print(f"Restored {round(num_restored/num_segmented_original * 100, 2)}% ({num_restored}/{num_segmented_original})")
    print(f'Validation done in: {round((perf_counter() - start_validation_time), 2)} sec.')
