import numpy as np
import os
from scipy.io import savemat, loadmat
import csv

#  Loads all the mats inside a list as numpy arrays
def load_mats(directory, filenames):
  columns = range(78, 82) # EMG columns for non-zero check

  mats = []
  filenames_filtered = []

  for i in filenames:
    aux = loadmat(directory+i)
    # Key is not always the same, accessing it by index num    
    mat = aux[list(aux)[3]]    
    
    if np.all(mat[0][columns] != np.array([0,0,0,0])):
      mats.append(mat)
      filenames_filtered.append(i)
    else:
      print(i)
  
  return mats, filenames_filtered


# All the behaviours are combined as "protective"
def merge_behaviour_labels(mat):
  NUM_EXERCISES = 7 

  binary_labels = np.zeros((mat.shape[0], NUM_EXERCISES))
  
  for idx, (row) in enumerate(mat): # K is a row, k[x] access columns
    for ex in range(NUM_EXERCISES):
      tmpROW = [row[83+ex], row[90+ex], row[97+ex], row[104+ex]]
      count0 = sum([1 for x in tmpROW if x == 0])
      count2 = sum([1 for x in tmpROW if x == 2])
      if count0 == 4:
        binary_labels[idx, ex] = 0
      elif count0 == 3:
        if ex == 0 and count2 == 1:
          binary_labels[idx, ex] = 1
        else:
          binary_labels[idx, ex] = 0
      else:
        binary_labels[idx, ex] = 1

  return binary_labels
    
# Transform the 4 raters 6-class ternary labelling to a binary label
def transform_labels(mats, names):
  
  label_dict = {}

  for idx, name in enumerate(names):  
        data = merge_behaviour_labels(mats[idx])
        for col_idx in range(data.shape[1]):
            col_name = f"{name}_{col_idx}"
            label_dict[col_name] = data[:, col_idx]
    
  return label_dict


if __name__ == '__main__':
    
    dirname = os.path.dirname(__file__)
    
    data_dir = "" ## SET RAW DATA DIRECTORY HERE
    folders = os.listdir(data_dir)

    c_mat_files = os.listdir(data_dir+'/randomised C')
    p_mat_files = os.listdir(data_dir+'/randomised P') 

    # Sorting to ensure always the same order
    c_mat_files.sort()
    p_mat_files.sort()

    # Loads .mat files
    p_mats, p_mat_files_filtered = load_mats(data_dir+'/randomised P/', p_mat_files)
    c_mats, c_mat_files_filtered = load_mats(data_dir+'/randomised C/', c_mat_files)

    # Binary labels
    p_labels_dict = transform_labels(p_mats, p_mat_files_filtered)
    c_labels_dict = transform_labels(c_mats, c_mat_files_filtered)      
            
    savemat('p_labels.mat', p_labels_dict)
    savemat('c_labels.mat', c_labels_dict)
            
    def save_dict_as_csv(dictionary, file_path): 
      with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        header = dictionary.keys()
        writer.writerow(header)
        max_len = max(len(dictionary[key]) for key in header)
        for i in range(max_len):
          row = [dictionary[key][i] if i < len(dictionary[key]) else '' for key in header]
          writer.writerow(row)
        
    save_dict_as_csv(p_labels_dict, 'p_labels.csv')
    save_dict_as_csv(c_labels_dict, 'c_labels.csv')