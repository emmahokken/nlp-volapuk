def consolidate():
    with open('latinlangs_emma.txt', 'r') as a, open('latinlangs_adriaan.txt', 'r') as b, open('latinlangs_toby.txt', 'r') as c, open('latinlangs.txt', 'w') as f:
        emma = set()
        adriaan = set()
        toby = set()
        total = set()
        for lan in a:
            emma.add(lan)
            total.add(lan)
        for lan in b:
            adriaan.add(lan)
            total.add(lan)
        for lan in c:
            toby.add(lan)
            total.add(lan)

        for lan in total:
            c=0
            for s in [emma, adriaan, toby]:
                if lan in s:
                    c+=1
            if c > 2:
                f.write(lan)

def latin_languages():
    lans = set()
    with open('latinlangs.txt', 'r') as f:
        for line in f:
            lans.add(line.rstrip())
    return lans

if __name__ == '__main__':
    consolidate()
    print(latin_languages())