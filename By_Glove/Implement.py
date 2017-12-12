from data_helpers import *
from embedding_loader import *

# 1-1-Save.
loaded_data = load_data_and_labels_another()
dictionary1 = {}
dictionary1['x'] = loaded_data[0]
dictionary1['y'] = loaded_data[1]
f = open("data/data_pickling", 'wb')
pickle.dump(dictionary1,f)
f.close()


# 1-2- Implement.
main("data")

# 1-3- Implement.
word_id_convert("data")


