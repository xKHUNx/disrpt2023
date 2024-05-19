import json
import stanza

"""
Convert custom dataset to HITS at DISRPT 2023 JSON format.
"""

DNAME = "eng.sdrt.malaysia_hansard"
OUTPUT_JSON_PATH = r"eng.sdrt.malaysia_hansard_test.json"
json_paths = [
    # r"/mnt/c/Users/user/Documents/Monash/Research/hansard_exploration/data/hansard_jsons/15_Parlimen_Kelima_Belas/02_Penggal_Kedua/02_Mesyuarat_Kedua/DR-06062023.json"
    r"/mnt/c/Users/user/Documents/Monash/Research/hansard_exploration/src/hansard_preprocessing/test0606.json"
    # r"C:\Users\user\Documents\Monash\Research\hansard_exploration\data\hansard_jsons\15_Parlimen_Kelima_Belas\02_Penggal_Kedua\02_Mesyuarat_Kedua\DR-06062023.json"
]


# Instantiate an instance of Stanza NLP pipeline which can be reused
nlp = stanza.Pipeline(lang="en", processors="tokenize, mwt, pos, lemma, depparse")


def get_token_pos_tags(sent):
    """Run a sentence through the Stanza pipleline to obtain
    the tokens and Part-of-Speech (POS) tags.

    Args:
        sent (str): The sentence to be tokenized and POS-tagged.

    Returns:
        list: A list of strings, containing the tokens for
            each sentence
        list: A list containing a list for each token with
            the following content:
            1. lemmatized token
            2. universal POS tags (UPOS)
            3. treebank-specific POS tags (XPOS)
            4. universal morphological features (UFeats)
            5. syntactic head of each word in a sentence
            6. dependency relation between the two words that
               are accessible
    """

    # Run the sentence through the pipeline
    doc = nlp(sent)
     
    doc_sents = []
    doc_sent_token_features = []
    for word in doc.sentences[0].words:
        doc_sent_token_features.append([
            word.lemma,
            word.upos,
            word.xpos,
            word.feats,
            word.head,
            word.deprel
        ])
        doc_sents.append(word.text)

    return doc_sents, doc_sent_token_features

def write_dicts_to_file(dict_list, file_path):
    with open(file_path, "w") as file:
        for dictionary in dict_list:
            json.dump(dictionary, file)
            file.write("\n")


for json_path in json_paths:

    # Load custom data JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Each line contain a dictionary
    output_json_lines = []

    # Process each agenda (conversation)
    for agenda in data["agendas"]:
        output_json_line = {}
        output_json_line["doc_id"] = agenda["title"]
        output_json_line["dname"] = DNAME
        output_json_line["doc_sents"] = []
        output_json_line["doc_sent_token_features"] = []
        # Process each sentence
        for sent in agenda["translated_sentences"]:
            tok, pos = get_token_pos_tags(sent["translated_content"])
            output_json_line["doc_sents"].append(tok)
            output_json_line["doc_sent_token_features"].append(pos)
        output_json_lines.append(output_json_line)
        
    # Save converted data
    write_dicts_to_file(output_json_lines, OUTPUT_JSON_PATH)

