from datareader import DataSet

data = DataSet()

with open("./data/wili-2018/x_train.txt", 'r') as f:
    fullthing = f.read()

alphas=0
nonalphas=0
chars = set(fullthing)
for c in chars:
    if c.isalpha():
        alphas+=1
    else:
        nonalphas+=1

print("For the full training data:")
print("Number of paragraphs:", len(data.paragraphs))
print("Number of unique words:", len(set(fullthing.split())))
print("Number of unique characters:", len(set(fullthing)))
print(f"Of which {alphas} alphabetical and {nonalphas} a form of interpunction")
print("Number of languages:", len(data.languages))


dutch = []
for _ in range(500):
    dutch.append(data.get_next_batch(1, ['nld'])[0][0])
fullthing = "".join(dutch)[:-1]

alphas=0
nonalphas=0
chars = set(fullthing)
for c in chars:
    if c.isalpha():
        alphas+=1
    else:
        nonalphas+=1
print("\n\nFor dutch paragraphs only:")
print("Number of paragraphs:", len(fullthing.split('\n')))
print("Number of unique words:", len(set(fullthing.split())))
print("Number of unique characters:", len(set(fullthing)))
print(f"Of which {alphas} alphabetical and {nonalphas} a form of interpunction")
print("Number of languages:", len(data.languages))

print("\n\nAll dutch characters:")
print(set(fullthing))