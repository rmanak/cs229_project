import re
import json
import sys

symbol_pattern = re.compile(r"^(.*?) \(([A-Z]+(?:\.[A-Z]+)?)\)")

# Regular expression pattern to extract the required information
expectation_re = re.compile(
    r"EPS of \$([\d.]+) (beats|misses) by \$([-\d.]+) \| Revenue of \$(\d+.\d+)([MB]) \(([-\d.]+)% Y/Y\) (beats|misses) by \$([\d.]+[MB])"
)

def is_paragraph_break(current_line, next_line):
    """
    Checks if there should be a paragraph break between the current line and the next line.
    
    :param current_line: The current line being analyzed.
    :param next_line: The next line after the current line.
    :return: True if there's a paragraph break, False otherwise.
    """
    # Check if the current line starts with a number or lowercase letter and ends with a period
    if current_line and (current_line[0].isdigit() or current_line[0].islower()) and current_line.rstrip().endswith('.'):
        # Split the next line into words and check if it starts with a capital letter and is at least 5 words long
        next_line_words = next_line.split()
        if next_line and next_line[0].isupper() and len(next_line_words) >= 5:
            return True
    return False

def is_speaker_line(line):
    """
    Checks if a line likely represents the name of a speaker.
    Criteria: No more than 4 words, all of which begin with upper-case letters.
    
    :param line: The line to check.
    :return: True if the line matches the criteria, False otherwise.
    """
    # Trim the line to remove leading and trailing whitespace
    trimmed_line = line.strip()

    # Split the line into words
    words = trimmed_line.split()

    # Check if the line has no more than 4 words
    if len(words) > 4:
        return False
    
    # Check if all words start with an upper-case letter
    if all(word[0].isupper() for word in words):
        # If last word ends in a period, we probably captured the end of a short sentence. Don't count it as a speaker.
        if words[-1].endswith('.'):
            return False
        # Use regex to further ensure each word starts with an uppercase letter followed by any characters
        # This will filter out lines that consist of uppercase abbreviations or single letters
        return all(re.match('^[A-Z][a-z]*', word) for word in words)
    
    return False

def extract_paragraphs(file):
    lines = []
    paragraphs = []
    previous_line = ""
    line = ""
    speaker = "UNKNOWN"
    before_transcript = True
    participants = {}
    def paragraphs_update(paragraphs, speaker, lines):
        if lines:
            paragraphs.append({
                "speaker": speaker,
                "text": "".join(lines).replace("\n", " ").strip(),
            })
            lines.clear()
        return
    while True:
        tmp_previous_line= line
        line = file.readline()
        if not line: break
        if not line.strip(): continue
        previous_line = tmp_previous_line
        if re.match(r"^https://seekingalpha.com/article/", line):
            for i in range(7):
                _ = file.readline()
            continue
        if before_transcript:
            # Handle cases where we have both "Company Participants" and "Conference Call Participants" in a transcript.
            if line.strip().endswith(" Participants"):
                continue
            match = re.match(r"^(.*?) +- +(.*?)$", line)
            if not match:
                before_transcript = False
            else:
                participants[match.group(1)] = match.group(2)
                continue
        if is_speaker_line(line):
            paragraphs_update(paragraphs, speaker, lines)
            speaker = line.strip()
            continue
        if is_paragraph_break(previous_line, line):
            paragraphs_update(paragraphs, speaker, lines)
        lines.append(line)
    return participants, paragraphs
    

outputs = {}
for file_path in sys.argv[1:]:
    output = {}
    output["expectation_results"]  = {"eps": {}, "revenue": {}}
    symbol = None
    try:
        with open(file_path, "r") as file:
            while True:
                line = file.readline()
                if not line: break
                if line.strip() == "Company Participants":
                    participants, paragraphs = extract_paragraphs(file)
                    output["participants"] = participants
                    output["paragraphs"] = paragraphs
                    continue
                if symbol is None and len(line.strip()) > 32:
                    match = symbol_pattern.search(line)
                    if match:
                        output["company_name"] = match.group(1)
                        symbol = match.group(2)
                match = expectation_re.search(line)
                if match:
                    # Extracting captured values
                    (
                        eps_value,
                        eps_beats_misses,
                        eps_beats_by,
                        revenue_value,
                        revenue_suffix,
                        revenue_yoy,
                        revenue_beats_missed,
                        revenue_beats_by,
                    ) = match.groups()

                    # Add extracted data to the expectation_results dictionary
                    output["expectation_results"]["eps"]["value"] = eps_value
                    output["expectation_results"]["eps"]["beats_by"] = (
                        eps_beats_by
                        if eps_beats_misses == "beats"
                        else "-" + eps_beats_by
                    )
                    output["expectation_results"]["revenue"]["value"] = (
                        revenue_value + revenue_suffix
                    )
                    output["expectation_results"]["revenue"]["YoY_percent"] = revenue_yoy
                    output["expectation_results"]["revenue"]["beats_by"] = (
                        revenue_beats_by
                        if revenue_beats_missed == "beats"
                        else "-" + revenue_beats_by
                    )
                        
    except FileNotFoundError:
        print(f"The file {file_path} was not found.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
    if symbol == None:
        print(f"The file {file_path} does not have a symbol.", file=sys.stderr)
        continue
    outputs[symbol] = output

# Convert the structured_data dictionary to a JSON string
json_data = json.dumps(outputs, indent=2)

# Output the JSON string
print(json_data)
