import json
import os
import pickle
import re
from time import perf_counter

import pynini
from alignment import (
    create_symbol_table,
    get_string_alignment,
    get_word_segments,
    indexed_map_to_output,
    remove, _get_aligned_index
)
from joblib import Parallel, delayed
from nemo_text_processing.text_normalization.normalize import Normalizer
from tqdm import tqdm
import string

def remove_punctuation(text, remove_spaces=True, do_lower=True, exclude=None):
    all_punct_marks = string.punctuation

    if exclude is not None:
        for p in exclude:
            all_punct_marks = all_punct_marks.replace(p, "")

        # a weird bug where commas is getting deleted when dash is present in the list of punct marks
        all_punct_marks = all_punct_marks.replace("-", "")
    text = re.sub("[" + all_punct_marks + "]", " ", text)

    if exclude and "-" not in exclude:
        text = text.replace("-", " ")

    text = re.sub(r" +", " ", text)
    if remove_spaces:
        text = text.replace(" ", "").replace("\u00A0", "").strip()

    if do_lower:
        text = text.lower()
    return text.strip()

NA = "n/a"

def clean(text):
    text = (
        text.replace("-", " ").replace(":", " ")
        # .replace(" .", ".")
        # .replace(" ,", ",")
        # .replace(" ?", "?")
        # .replace("  ", " ")
    )
    return text


def build_output_raw_map(alignment, output_text, text):
    # get TN alignment
    indices = get_word_segments(text)
    output_raw_map = []

    for i, x in enumerate(indices):
        try:
            start, end = indexed_map_to_output(start=x[0], end=x[1], alignment=alignment)
        except:
            print(f"{key} -- error")
            return None

    output_raw_map.append([output_text[start:end], text[x[0] : x[1]]])
    return output_raw_map


def process_with_normalizer(
    item, key, normalizer, offset=5, output_raw_map=None, segmented_field="segmented", raw_text_field="raw_text"
):
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

                processed_raw = clean(normalizer.normalize(restored_raw).lower())
                processed_segment = clean(segment.lower())
                processed_restored = clean(restored_norm.lower())

                # print(f"RAW :{processed_raw}")
                # print(f"SEGM:{processed_segment}")
                # print(f"REST:{processed_restored}")
                # import pdb; pdb.set_trace()
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


def build_output_raw_map(alignment, output_text, text):
    indices = get_word_segments(text)
    output_raw_map = []

    for i, x in enumerate(indices):
        try:
            start, end = indexed_map_to_output(start=x[0], end=x[1], alignment=alignment)
        except:
            print(f"{key} -- error")
            return None

        output_raw_map.append([output_text[start:end], text[x[0] : x[1]]])
    return output_raw_map


def get_raw_text_from_alignment(alignment, alignment_start_idx=0, alignment_end_idx=None):
    if alignment_end_idx is None:
        alignment_end_idx = len(alignment)

    return "".join(list(map(remove, [x[0] for x in alignment[alignment_start_idx : alignment_end_idx + 1]])))


def process(item, key, normalizer, use_cache=True, verbose=False, with_normalizer=True):
    start_time = perf_counter()
    restored = {}
    text = item["raw_text"]

    if "segmented" not in item:
        return (key, restored)

    segmented = item["segmented"]
    cached_alignment = f"alignment_{key}.p"

    fst = pynini.compose(normalizer.tagger.fst, normalizer.verbalizer.fst)
    if os.path.exists(cached_alignment) and use_cache:
        alignment = pickle.load(open(f"alignment_{key}.p", "rb"))
    else:
        print(f"generating alignments {key}")
        table = create_symbol_table()
        alignment, _ = get_string_alignment(fst=fst, input_text=text, symbol_table=table)
        pickle.dump(alignment, open(f"alignment_{key}.p", "wb"))
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
        # if "hi, welcome to thecube virtual." in segment_clean:
        #     import pdb; pdb.set_trace()
        if segment_clean in segmented_result_clean:
            segment_start_idx = max(segmented_result_clean.index(segment_clean) - 1, 0)

            alignment_start_idx = segmented_indices[segment_start_idx]
            segment_end_idx = min(segment_start_idx + len(segment), len(segmented_indices) - 1)
            alignment_end_idx = segmented_indices[segment_end_idx]

            # alignment_start_idx  = _get_aligned_index(alignment, segment_start_idx)
            # alignment_end_idx = _get_aligned_index(alignment, segment_end_idx)

            raw_text = "".join(list(map(remove, [x[0] for x in alignment[alignment_start_idx: alignment_end_idx + 1]])))
            restored[id] = raw_text
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

    if with_normalizer:
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
                restored[failed[i][0]] = failed_restored
                if verbose:
                    print("=" * 40)
                    print(f"RAW : {raw_text_}")
                    print(f"SEGM: {failed[i][1]}")
                    print("=" * 40)

    for i in range(len(segmented)):
        if i in restored:
            item["misc"][i]["text_pc"] = restored[i]

    result = [x for x in item["misc"] if "text_pc" in x]
    if verbose:
        print(
            f"{key} -- found: {len(result)} out of {len(item['segmented'])} ({round(len(result) / len(item['segmented']) * 100, 1)}%)"
        )
    print(
        f'Restoration {key}: {round((perf_counter() - start_time), 2)} sec -- {len(item["segmented"])} -- with normalizer: {len(failed)}'
    )
    return result


if __name__ == "__main__":
    raw_text = ""
    norm_text = ""
    data_dir = "/media/ebakhturina/DATA/mlops_data/pc_retained"
    raw_manifest = f"{data_dir}/raw.json"
    segmented_manifest = f"{data_dir}/segmented.json"

    # for every audio file store "raw" and "segmented" samples
    # data["Am4BKyvYBgY"].keys()
    # >> dict_keys(['raw_text', 'segmented'])

    data = {}
    # read raw manifest
    with open(raw_manifest, "r") as f_in:
        for line in f_in:
            line = json.loads(line)
            text = re.sub(r" +", " ", line["text"])
            audio = line["audio_filepath"].split("/")[-1].replace(".wav", "")
            data[audio] = {"raw_text": text}

    segmented = {}
    misc_data = {}
    # read manifest after segmentation
    with open(segmented_manifest, "r") as f_in:
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
            print(f"{audio} from {segmented_manifest} is missing in the {raw_manifest}")
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
    cache_dir = "cache_dir"
    lang = "en"
    normalizer = Normalizer(input_case="cased", cache_dir=cache_dir, overwrite_cache=False, lang=lang)

    # result = []
    # start_overall_time = perf_counter()
    # for key, item in tqdm(data.items()):
    #     if key == 'AnYuZcVmFeQ':
    #         print(f"processing {key}")
    #         result.append(process(item, key, normalizer, use_cache=True, verbose=False, with_normalizer=False))
    #
    # print(f'ALL restored: {round((perf_counter() - start_overall_time)/60, 2)} min.')

    use_cache = True
    verbose = False
    with_normalizer = False

    start_time = perf_counter()
    result = Parallel(n_jobs=10)(
        delayed(process)(item, key, normalizer, use_cache=use_cache, verbose=verbose, with_normalizer=with_normalizer)
        for key, item in tqdm(data.items())
    )
    print(f'Restoration parallel: {round((perf_counter() - start_time), 2)} sec.')
    print(f"use_cache: {use_cache}, verbose: {verbose}, with_normalizer: {with_normalizer}")


    start_validation_time = perf_counter()
    restored_lines = []
    for audio_segments in result:
        for line in audio_segments:
            restored_lines.append(line["text_pc"])

    # this is need to validate alignments
    normalizer_prediction = normalizer.normalize_list(
        restored_lines,
        verbose=False,
        punct_pre_process=False,
        punct_post_process=False,
        batch_size=300,
        n_jobs=10,
    )

    drop = []
    drop_text_error = 0
    num_restored = 0
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    idx = 0
    with open(f"{output_dir}/{os.path.basename(segmented_manifest).replace('.json', '_restored.json')}", "w") as f_out:
        for audio_segments in result:
            for line in audio_segments:
                normalized_pc = normalizer_prediction[idx]
                if remove_punctuation(line["text"]) != remove_punctuation(normalized_pc):
                    if len(line["text"]) < len(normalized_pc) and line["text"].lower() == normalized_pc[1:].lower():
                        drop_text_error += 1
                    else:
                        drop.append((line["text"], normalized_pc))
                else:
                    f_out.write(json.dumps(line, ensure_ascii=False) + '\n')
                    num_restored += 1
                idx += 1

    print(f"Dropped {len(drop)}, dropped_text_error: {drop_text_error}")
    if verbose:
        for d in drop:
            print(f'TEXT: {d[0]}')
            print(f'RESTORED: {d[1]}')
            print("-" * 40)


    num_segmented_original = sum([len(item['segmented']) for item in data.values()])
    print(f"Restored {round(num_restored/num_segmented_original * 100, 2)}% ({num_restored}/{num_segmented_original})")
    print(f'Validation done in: {round((perf_counter() - start_validation_time), 2)} sec.')