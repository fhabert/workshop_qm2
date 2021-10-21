import find_k

to_test = find_k.x_train_sigmoid[2]
k_value = find_k.elbow_point
validation_labels = find_k.y_train

def get_diabetes_answer(unknown, k, labels):
    distances = []
    l = len(find_k.x_train_sigmoid)
    for i in range(l):
        distances.append([find_k.distance(unknown, find_k.x_train_sigmoid[i]), i])
    distances.sort()
    num_good = 0
    num_bad = 0
    looked_at = distances[:k]
    for i in range(0, len(looked_at)):
        if labels.iloc[looked_at[i][1]] == 1:
            num_good += 1
        elif labels.iloc[looked_at[i][1]] == 0:
            num_bad += 1
    if num_good > num_bad:
        return 1
    return 0

result = get_diabetes_answer(to_test, k_value, validation_labels)
print(result)
# with a 0.7 margin of error