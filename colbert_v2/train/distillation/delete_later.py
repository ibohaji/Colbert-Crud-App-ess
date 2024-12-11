import json

mapping = {}
new_data = {}

with open('colbert_v2/datasets/scifact/corpus.jsonl') as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        mapping[data['_id']] = idx
        new_data[idx] = data


with open('mapping_collection_ids.json', 'w') as f:
    f.write(json.dumps(mapping, indent=4))

with open('collection_data_altered.json', 'w') as f:
    f.write(json.dumps(new_data, indent=4))


def process_json_file(input_path, output_path, mapping):
    """Processes an NDJSON file to update IDs using a mapping dictionary.

    Args:
        input_path (str): Path to the input NDJSON file.
        output_path (str): Path to the output NDJSON file.
        mapping (dict): Dictionary mapping old IDs to new IDs.
    """
    with open(input_path) as fin, open(output_path, 'w') as fout:
        for line_number, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                # Parse the JSON array
                data = json.loads(line)

                # Validate the structure
                if not isinstance(data, list) or len(data) != 2:
                    print(f"Warning: Line {line_number} is not a valid [key, value] pair. Skipping.")
                    continue

                key, scores_ids = data

                # Ensure 'scores_ids' is a list of lists
                if not isinstance(scores_ids, list):
                    print(f"Warning: Line {line_number} 'scores_ids' is not a list. Skipping.")
                    continue

                # Update the IDs using the mapping
                updated_scores_ids = []
                for pair in scores_ids:
                    if not isinstance(pair, list) or len(pair) != 2:
                        print(f"Warning: Line {line_number} has an invalid pair format: {pair}. Skipping this pair.")
                        continue
                    score, old_id = pair


                    try:

                        new_id = int(mapping[old_id])
                    except KeyError:
                        print(f"Warning: Line {line_number} ID '{old_id}' not found in mapping. Keeping original ID.")
                        raise KeyError

                    if old_id not in mapping:
                        print(f"Warning: Line {line_number} ID '{old_id}' not found in mapping. Keeping original ID.")
                    updated_scores_ids.append([score, new_id])

                # Create the updated entry
                updated_entry = [key, updated_scores_ids]

                # Write the updated entry to the output file
                json.dump(updated_entry, fout)
                fout.write('\n')  # Ensure each JSON array is on a new line

            except json.JSONDecodeError as e:
                print(f"Error: Failed to decode JSON on line {line_number}: {e}. Skipping this line.")
            except Exception as e:
                print(f"Unexpected error on line {line_number}: {e}. Skipping this line.")

process_json_file('altered_distillation.json', 'altered_distillations_scores.json', mapping)
