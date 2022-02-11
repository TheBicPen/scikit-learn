from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(class_weight={1: 10, 0: 1})
rfc.fit([[0, 0, 1], [1, 0, 1]], [0, 0])
