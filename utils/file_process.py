import os
import json
import pandas as pd


def excel_reader(path):
    records = []
    if os.path.isfile(path):
        df = pd.read_excel(path)
        for row in df.itertuples(index=False):
            records.append(row.asdict())
    return records


def jsonl_writer(path, data):
    if path.endswith("jsonl"):
        writer = open(path, 'w', encoding='utf-8')
        for item in data:
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")


def jsonl_reader(path):
    res = []
    if path.endswith("jsonl"):
        with open(path, 'r', encoding='utf-8') as reader:
            for line in reader:
                if line is not None:
                    line = line.replace("\\\n", "\n")
                    data = json.loads(line)
                    res.append(data)
    return res


def json_reader(path):
    data = ""
    if path.endswith("json"):
        with open(path, 'r', encoding='utf-8') as reader:
            data = json.load(reader)
    return data


def json_writer(path, data):
    with open(path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False)


def txt_writer(path, data):
    writer = open(path, 'w', encoding='utf-8')
    for item in data:
        writer.write(item + "\n")


def txt_reader(path):
    cases = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            cases.append(line.strip())
    return cases


def excel_writer(path, data):
    if path.endswith(".xlsx"):
        df = pd.DataFrame(data)
        # 保存到 Excel 文件
        df.to_excel(path, index=False)


def xlsx_to_csv_pd(file_in, file_out):
    data_xls = pd.read_excel(file_in,index_col=0)
    data_xls.to_csv(file_out)

def csv_reader(file_in):
    result = []
    if file_in.endswith("csv"):
        result = pd.read_csv(file_in)
    return result



if __name__ == "__main__":
    file_in = "/Users/chendongdong/Work/dataset/quality_evaluation/丹鸟-更改地址/all.xlsx"
    file_out = "/Users/chendongdong/Work/dataset/quality_evaluation/丹鸟-更改地址/all.csv"
    xlsx_to_csv_pd(file_in, file_out)