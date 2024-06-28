import openpyxl as xl
import csv
import jinja2
import pathlib
import random
from tqdm import tqdm

with pathlib.Path("template.jinja2").open() as f:
    prompt_template = jinja2.Template(f.read())

labels = [
    {"label": 0, "answer": "Not Satisfactory"},
    {"label": 1, "answer": "Satisfactory"},
]

questions = []
with open('Questions.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for index, row in enumerate(reader):
        if index != 0:
            questions.extend(row)


wb = xl.load_workbook("health_articles_data.xlsx")

for index in list(range(1)):
    # print("Active sheet: ", wb.active)
    wb._active_sheet_index = index
    sheet = wb.active

    dataDict = {}
    for row in tqdm(range(990, 1000)):
        data = list(sheet.iter_cols(1, sheet.max_column))
        example = data[21][row].value
        answer = data[18][row].value
        split = data[-1][row].value
        dataDict[row] = {"example": example, "question": questions[index], "answer": answer, "split": split}

        # print(f"Q:{questions[index]}\nEx:{example}\nAns:{answer}\nSplit:{split}\n\n")

    test_list = [index for index, data in dataDict.items() if data['split'] == 'test']
    train_list = [index for index, data in dataDict.items() if data['split'] == 'train']

    for test_index in test_list[:1]:

        train_index_1, train_index_2 = random.sample(train_list, 2) # train_list[:2]

        train_list.remove(train_index_1)
        train_list.remove(train_index_2)

        train_data_1, train_data_2 = dataDict[train_index_1], dataDict[train_index_2]

        test_data = dataDict[test_index]

        # print(f"{train_data_1}")
        # print(f"{train_data_2}")
        # print(f"{test_data}")

        examples = [
            {"context": train_data_1["example"], "label": train_data_1["answer"]}, # if == "" then 1 or 0
            {"context": train_data_2["example"], "label": train_data_2["answer"]},
        ]

        test = test_data["example"]

        prompt = prompt_template.render(
            labels=labels,
            examples=examples,
            test=test,
            question=questions[index]
        )

        print(prompt)

