



def get_cls(rating):
    lbl=[]
    for i in range(len(rating)):
        if rating[i] >=2.5:
            lbl.append(0)
        else:
            lbl.append(1)

    return lbl