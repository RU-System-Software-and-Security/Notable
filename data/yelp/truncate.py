import csv

lines = []

with open("train.tsv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for line in reader:
        if len(line[0].split(" ")) < 200:
            lines.append(line)


with open("create_anchor.tsv", "w") as w:
    writer = csv.writer(w, delimiter='\t')
    idx = 0
    for line in lines:
        writer.writerow(line)
        if idx >= 5000:
            break
        idx += 1


