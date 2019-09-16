from datareader import DataSet
import sys

data = DataSet()

done = set()
latin = set()
for _ in range(len(data.paragraphs)):
    batch = data.get_next_batch(1)
    if batch[1][0] in done:
        continue
    print(batch[0][0])
    print(batch[1][0])
    print(len(done), '/', len(data.languages))
    i = 'c'
    while i[0].lower() not in 'yn':
        print("latin? (y/n)")
        i = sys.stdin.readline()
    if i[0].lower() == 'y':
        latin.add(batch[1][0])
    done.add(batch[1][0])

with open("latinlangs.txt", 'w') as f:
    for lan in latin:
        f.write(lan)
        f.write('\n')