import dill

def serialize(object, filename):
    with open("./tmp/" + filename + ".pickle", "wb") as file:
        dill.dump(object, file, protocol=dill.HIGHEST_PROTOCOL, fix_imports=True)


def deserialize(filename):
    with open("./tmp/" + filename + ".pickle", "rb") as file:
        return dill.load(file)
