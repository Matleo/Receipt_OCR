from glob import glob
def main(folder1, folder2, fontSizes):
    print("Searching Folder 1")
    # find file names in folder1
    bogus1 = []
    noBogus1 = []
    for fontSize in fontSizes:
        fontFolder = folder1+"/"+fontSize
        classes = glob(fontFolder+"/*/")
        for aClass in classes:
            noBogusFiles = glob(aClass+"*.png")
            for file in noBogusFiles:
                file = file.replace("\\","/")
                file = file.replace(folder1+"/","")
                noBogus1.append(file)
            bogusFiles = glob(aClass+"bogus/*.png")
            for file in bogusFiles:
                file = file.replace("\\","/")
                file = file.replace(folder1+"/","")
                file = file.replace("/bogus","")
                bogus1.append(file)

    #find file names in folder2
    print("Searching Folder 2")
    bogus2 = []
    noBogus2 = []
    for fontSize in fontSizes:
        fontFolder = folder2+"/"+fontSize
        classes = glob(fontFolder+"/*/")
        for aClass in classes:
            noBogusFiles = glob(aClass+"*.png")
            for file in noBogusFiles:
                file = file.replace("\\","/")
                file = file.replace(folder2+"/","")
                noBogus2.append(file)
            bogusFiles = glob(aClass+"bogus/*.png")
            for file in bogusFiles:
                file = file.replace("\\","/")
                file = file.replace(folder2+"/","")
                file = file.replace("/bogus","")
                bogus2.append(file)

    # compare file names:
    print("Comparing File Names:")
    print("--- bogus1: ")
    for aBog in bogus1:
        if aBog not in bogus2:
            if aBog in noBogus2:
                print(aBog + " is bogus in folder1 and noBogus in folder2")
            else:
                print(aBog + " is only present in folder1(bogus)")

    print("--- bogus2: ")
    for aBog in bogus2:
        if aBog not in bogus1:
            if aBog in noBogus1:
                print(aBog + " is bogus in folder2 and noBogus in folder1")
            else:
                print(aBog + " is only present in folder2(bogus)")

    print("--- noBogus1:")
    for i in range(len(noBogus1)):
        aNoBog = noBogus1[i]
        if aNoBog not in noBogus2:
            if aNoBog in bogus2:
                print(aNoBog + " is noBogus in folder1 and bogus in folder2")
            else:
                print(aNoBog + " is only present in folder1(noBogus)")

    print("--- noBogus2:")
    for i in range(len(noBogus2)):
        aNoBog = noBogus2[i]
        if aNoBog not in noBogus1:
            if aNoBog in bogus1:
                print(aNoBog + " is noBogus in folder2 and bogus in folder1")
            else:
                print(aNoBog + " is only present in folder2(noBogus)")
    print("Done")

if __name__=="__main__":
    folder1 = "../assets/Denns/raw"
    folder2 = "../assets/DennsNew/raw"
    fontSizes = ["10","12"]
    main(folder1, folder2, fontSizes)