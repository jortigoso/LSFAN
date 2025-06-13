import pandas as pd
import numpy as np
import os
from scipy.io import savemat, loadmat

# Test for sequence number
sqs = 0

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
      
  print('-----')
  return mats, filenames_filtered

def sliding_window(matrix, window_size, step, columns):
  selected_columns = matrix[:, columns]
  num_windows = (selected_columns.shape[0] + step - 1) // step
  windows = []
  for i in range(num_windows):
    start = i * step
    end = min(start + window_size, selected_columns.shape[0])
    window = selected_columns[start:end, :]
    padding = np.zeros((window_size - window.shape[0], window.shape[1]))
    window = np.concatenate((window, padding))
    windows.append(window)
  return windows

def sliding_window_np(matrix, window_size, step, columns):
    selected_columns = matrix[:, columns]
    num_windows = (selected_columns.shape[0] - window_size) // step + 1
    windows = []
    for i in range(num_windows):
        start = i * step
        end = start + window_size
        window = selected_columns[start:end, :]
        windows.append(window)
    return windows

# Lo sé... 
def frame_label_voting(arr, window_size, step):
    result = []
    sum_condition = lambda x: np.sum(x) >= 1
    for i in range(0, len(arr) - step + 1, step):
      window = arr[i:i + window_size]
      count = 0
      for row in window:
        if sum_condition(row) :
          count += 1
      result.append(int(count >= window.shape[0]/2))

    if len(arr) % step != 0:
      window = arr[-(len(arr) % step):]
      count = 0
      for row in window:
        if sum_condition(row) :
          count += 1
      result.append(int(count >= window.shape[0]/2))
    return result

def frame_label_voting_np(matrix, window_size, step):
    num_windows = (matrix.shape[0] - window_size) // step + 1
    sum_condition = lambda x: np.sum(x) >= 1
    labels = []
    for i in range(num_windows):
        start = i * step
        end = start + window_size
        window = matrix[start:end, :]

        count = 0
        for row in window:
          if(sum_condition(row)):
             count += 1   
        labels.append(int(count >= window.shape[0]/2))
    return labels

# Retruns the rows throughout a specific exercise
# extends, asumes one range
def get_exercise_samples(mat, exercise_idx):
  ls = list(mat[:,exercise_idx])
  try:
    first_index = ls.index(1)
    aux = ls[first_index:]
    last_index = first_index + aux.index(0)
  except:
    first_index = 0
    last_index = 0

  return range(first_index, last_index)


# Returns lists of paired data 'slice'+label for a specific exercise of a .mat
def get_exercise_frames(mat, mat_labels, exercise_idx):

  exercise_range = get_exercise_samples(mat, exercise_idx)

  data = mat[exercise_range]
  labels = mat_labels[exercise_range]
  
  window_size = 180 
  step_size = 45    
  # This should return a list of frames for  the selected exercise + each frame's label
  frames = sliding_window_np(data, window_size, step_size, columns = [*range(78, 82), *range(134, 160)])
  frames_labels = frame_label_voting_np(labels, window_size, step_size) 
  
  return frames, frames_labels


# Gets all the data slices (frames) and the corresponding labels for a list of .mats 
def get_mats_frames(mats, labels, mats_names, columns_names, exercise_names):

  frames = []
  frames_labels = []
  frames_ids = []
  
  frame_id = 0
  prev_id = mats_names[0].split('_')[0]

  for mat, mat_name in zip(mats, mats_names):
    if mat_name.split('_')[0] != prev_id:
      frame_id += 1
    prev_id = mat_name.split('_')[0]
    for i in exercise_names:
      label_columns = [f'{mat_name}_{idx}' for idx in range(7)]
      mat_labels = np.hstack([labels[col].T for col in label_columns])
      
      exercise_index = columns_names.index(i)
      exercise_frames, exercise_frame_labels = get_exercise_frames(mat, mat_labels, exercise_index)
    
      if(len(exercise_frames) != 0):    
        frames.append(exercise_frames)
        frames_labels.append(exercise_frame_labels)
        frames_ids.append([frame_id] * len(exercise_frames))
        
  return frames, frames_labels, frames_ids

def save_mats(frames, frames_labels, frames_ids, directory, filename_offset = 0, id_offset = 0):
  
  counter = 0
  for ex_frames, ex_labels, ex_ids in zip(frames, frames_labels, frames_ids):
    for frame, label, frame_id in zip(ex_frames, ex_labels, ex_ids):
      savemat(directory +'/'+ str(counter+filename_offset) +'.mat', {'id':frame_id+id_offset, 'data': frame, 'label': label})
      counter += 1
  return counter+1, max(frames_ids)[0]+1

if __name__ == "__main__":
    
    dirname = os.path.dirname(__file__)
    data_dir = "" # SET RAW DATA DIRECTORY HERE
    save_folder = "" # SET A DIRECTORY TO STORE INTIAL MATS
    folders = os.listdir(data_dir)

    columns_sheet = data_dir+'/FAME - Dataset Column Descriptions.xlsx'
    c_mat_files = os.listdir(data_dir+'/randomised C')
    p_mat_files = os.listdir(data_dir+'/randomised P') 

    # Sorting to ensure always the same order
    c_mat_files.sort()
    p_mat_files.sort()

    # Loads the .mat files that contain the binary labels generated with label_parser.py
    # FILES ARE LOADED AS NUMPY ARRAYS
    c_labels = loadmat('c_labels.mat')
    p_labels = loadmat('p_labels.mat')

    columns_description = pd.read_excel(columns_sheet)
    columns_names = list(columns_description['Description'])
    exercise_names = columns_names[110:134]

    exercise_names = ['One Leg Stand 1',
      'One Leg Stand 2',
      'One Leg Stand 3',
      'One Leg Stand 4',
      'One Leg Stand 5',
      'One Leg Stand 6',
      'Reach Forward 2',
      'Reach Forward 1',
      'Sit to Stand Instructed 1',
      'Sit to Stand Instructed 2',
      'Sit to Stand Instructed 3',
      'Stand to Sit Instructed 1',
      'Stand to Sit Instructed 2',
      'Stand to Sit Instructed 3',
      'Sit to Stand Not Instructed',
      'Stand to Sit Not Instructed',
      'Bend 2',
      'Bend 1',
      'Other Major: Bend to pick up',
      'Sit to Stand Instructed 4',
      'Stand to Sit Instructed 4']

    # Loads .mat files to get the sensor data
    p_mats, p_mat_files_filtered = load_mats(data_dir+'/randomised P/', p_mat_files)
    c_mats, c_mat_files_filtered = load_mats(data_dir+'/randomised C/', c_mat_files)

    # Generates lits of labelled data 'slices'
    p_frames, p_frames_labels, p_frames_ids = get_mats_frames(p_mats, p_labels, p_mat_files_filtered, columns_names, exercise_names)
    c_frames, c_frames_labels, c_frames_ids = get_mats_frames(c_mats, c_labels, c_mat_files_filtered , columns_names, exercise_names)


    c_offset, id_offset = save_mats(p_frames , p_frames_labels, p_frames_ids,save_folder)
    save_mats(c_frames, c_frames_labels, c_frames_ids, save_folder, c_offset, id_offset)

    print(f'Frames Healthy: {len(p_frames)}, labels: {len(p_frames_labels)}')
    print(f'Frames CP: {len(c_frames)}, labels: {len(c_frames_labels)}')