import openpyxl as xl
import csv
import jinja2
import pathlib
import random
from tqdm import tqdm

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

dataset = {}

def preprocess():

    with pathlib.Path("template.jinja2").open() as f:
        prompt_template = jinja2.Template(f.read())
    
    for index in list(range(4)):
        # print("Active sheet: ", wb.active)

        dataset[index] = []

        wb._active_sheet_index = index
        sheet = wb.active

        dataDict = {}
        for row in tqdm(range(0, sheet.max_row)):
            data = list(sheet.iter_cols(1, sheet.max_column))
            example = data[21][row].value
            answer = data[18][row].value
            split = data[-1][row].value
            dataDict[row] = {"example": example, "question": questions[index], "answer": answer, "split": split}

            # print(f"Q:{questions[index]}\nEx:{example}\nAns:{answer}\nSplit:{split}\n\n")

        test_list = [index for index, data in dataDict.items() if data['split'] == 'test']
        train_list = [index for index, data in dataDict.items() if data['split'] == 'train']

        for test_index in test_list:

            train_index_1, train_index_2 = random.sample(train_list, 2) # train_list[:2]

            train_list.remove(train_index_1)
            train_list.remove(train_index_2)

            train_data_1, train_data_2 = dataDict[train_index_1], dataDict[train_index_2]

            test_data = dataDict[test_index]

            # print(f"{train_data_1}")
            # print(f"{train_data_2}")
            # print(f"{test_data}")

            examples = [
                {"context": train_data_1["example"], "label": 1 if train_data_1["answer"] == "Satisfactory" else 0},
                {"context": train_data_2["example"], "label": 1 if train_data_2["answer"] == "Satisfactory" else 0},
            ]

            test = test_data["example"]

            prompt = prompt_template.render(
                labels=labels,
                examples=examples,
                test=test,
                question=questions[index]
            )

            dataset[index].append({"prompt": prompt, "target": test_data["answer"]})

    return dataset


def preprocess_training():

    with pathlib.Path("template_training.jinja2").open() as f:
        prompt_template = jinja2.Template(f.read())

    train_list, test_list = [], []

    for index in list(range(9)):
        # print("Active sheet: ", wb.active)
        wb._active_sheet_index = index
        sheet = wb.active

        local_train_list, local_test_list = [], []
        for row in tqdm(range(0, sheet.max_row)):
            data = list(sheet.iter_cols(1, sheet.max_column))
            context = data[21][row].value
            answer = data[18][row].value
            split = data[-1][row].value

            prompt = prompt_template.render(
                context=context,
                question=questions[index]
                # label=1 if answer == "Satisfactory" else 0
            )

            data = {"input": prompt, "label": 1 if answer == "Satisfactory" else 0}

            if split == 'train':
                local_train_list.append(data)
            else:
                local_test_list.append(data)

        perc = 0.5
        local_train_list = random.sample(local_train_list, int(len(local_train_list)*perc))
        local_test_list = random.sample(local_test_list, int(len(local_test_list)*perc))

        print(f"Completed {index+1} criteria with {len(local_train_list)} training dataset and "
              f"{len(local_test_list)} testing dataset")

        train_list.extend(local_train_list)
        test_list.extend(local_test_list)

    random.shuffle(train_list)
    random.shuffle(test_list)

    print(f"Total {len(train_list)} training dataset and " f"{len(test_list)} testing dataset")

    return train_list, test_list

if __name__ == '__main__':
    print(preprocess()[0]["prompt"])

