import pandas as pd
from scipy.io import loadmat, savemat
import numpy as np
import os

def load_mats(directory, filenames):
  columns = range(78, 82) # EMG columns for non-zero check
  mats = []
  filenames_filtered = []

  for filename in filenames:
    aux = loadmat(directory+filename)
    mat = aux[list(aux)[3]]    
    
    # Checks that the EMG columns have data since some
    # pain patients are empty
    if np.all(mat[0][columns] != np.array([0,0,0,0])):
      mats.append(mat)
      filenames_filtered.append(filename)
    else:
      print(f'{filename} has no EMGs')
      
  print('--------')
  return mats, filenames_filtered

def get_exercise_range(mat, exercise_idx):
    column = mat[:, exercise_idx]
    ones_indices = np.where(column == 1)[0]
    if len(ones_indices) == 0:
        return range(0)
    first_index = ones_indices[0]
    try:
        next_zero_index = np.where(column[first_index:] == 0)[0][0] + first_index
    except IndexError:
        next_zero_index = len(column)
    last_index = ones_indices[np.where(ones_indices < next_zero_index)[0][-1]] + 1
    return range(first_index, last_index)


def sliding_window(fragment, window, step, padding=True):
  if padding:
    frames, labels = sliding_window_padding(fragment, window, step)
    return frames, labels
  else:
    frames, labels = sliding_window_nopadding(fragment, window, step)
    return frames, labels
  
def sliding_window_padding(fragment, window_size, step):
  columns = [*range(78, 82), *range(134, 160)]
  rater_columns = [*range(82,110)]
  
  num_windows = (fragment.shape[0] + step - 1) // step

  windows = []
  labels = []
  for i in range(num_windows):
    start = i * step
    end = min(start + window_size, fragment.shape[0])
    window = fragment[start:end, :]

    window_data = window[:, columns]
    padding = np.zeros((window_size - window_data.shape[0], window_data.shape[1]))
    window_data = np.concatenate((window_data, padding))
    windows.append(window_data)

    window_raters = window[:, rater_columns]

    NUM_RATERS = 4
    NUM_EXERCISES = 6+1 
    
    binary_labels = np.zeros((window_raters.shape[0], NUM_RATERS))

    for idx, (row) in enumerate(window_raters):
      for rater in range(NUM_RATERS):
        aux = row[rater*NUM_EXERCISES : rater*NUM_EXERCISES + NUM_EXERCISES]
        binary_labels[idx][rater] = (1 if np.sum(aux) >= 1 else 0)

    num_rows, num_cols = binary_labels.shape
    half_rows_threshold = num_rows // 2
    count = 0
    for k in range(num_cols):
        # Count the number of rows with a value greater than 0
        num_rows_greater_than_zero = np.sum(binary_labels[:, k] > 0)

        # Check if the number of rows with a value greater than 0 is greater than or equal to half rows
        if num_rows_greater_than_zero >= half_rows_threshold:
            count += 1
    
    labels.append(1 if count >= num_cols // 2 else 0)

  return windows, labels

def sliding_window_nopadding(fragment, window_size, step):
    columns = [*range(78, 82), *range(134, 160)]
    rater_columns = [*range(82, 110)]

    num_windows = (fragment.shape[0] - window_size + step) // step

    windows = []
    labels = []
    for i in range(num_windows):
        start = i * step
        end = start + window_size
        if end > fragment.shape[0]:
            # Exclude windows that don't fit completely
            continue
        window = fragment[start:end, :]

        window_data = window[:, columns]
        windows.append(window_data)

        window_raters = window[:, rater_columns]
        # First treat all behaviours as one

        # Checks how many cells have a number
        # greater than the threshold (label can be 0, 1 or 2)
        type_condition = lambda x: sum(i >= 1 for i in x)

        NUM_RATERS = 4
        NUM_EXERCISES = 6 + 1

        binary_labels = np.zeros((window_raters.shape[0], NUM_RATERS))

        for idx, row in enumerate(window_raters):
            for rater in range(NUM_RATERS):
                aux = row[rater:rater + NUM_EXERCISES]
                binary_labels[idx][rater] = (1 if np.sum(aux) > 1 else 0)

        num_rows, num_cols = binary_labels.shape
        half_rows_threshold = num_rows // 2
        count = 0
        for k in range(num_cols):
            # Count the number of rows with a value greater than 0
            num_rows_greater_than_zero = np.sum(binary_labels[:, k] > 0)

            # Check if the number of rows with a value greater than 0 is greater than or equal to half rows
            if num_rows_greater_than_zero >= half_rows_threshold:
                count += 1

        labels.append(1 if count >= num_cols // 2 else 0)

    return windows, labels

      
def process_data(mats, mats_names, columns_names, exercise_names):

  print('..................')
  frames = []
  frames_labels = []
  frames_ids = []

  # Init ID
  frame_id = 0
  prev_id = mats_names[0].split('_')[0]
  
  for mat, mat_name in zip(mats, mats_names):
    # If the prefix changes increments ID
    if mat_name.split('_')[0] != prev_id:
      frame_id += 1
    prev_id = mat_name.split('_')[0]

    # Extract frames for each exercise separately
    for exercise in exercise_names:
      exercise_col_number = columns_names.index(exercise)

      exercise_range = get_exercise_range(mat, exercise_col_number)
      print(exercise_range)

      # Gets the rows of the selected exercise
      exercise_fragment = mat[exercise_range]

      window = 180 
      step = 45

      data_frame, label = sliding_window(exercise_fragment, window, step, padding=False)

      if(len(data_frame) > 0):
        frames.append(data_frame)
        frames_labels.append(label)
        frames_ids.append([frame_id] * len(data_frame))

  return frames, frames_labels, frames_ids

def save_mats(frames, frames_labels, frames_ids, directory, filename_offset = 0, id_offset = 0):
  
  counter = 0
  for ex_frames, ex_labels, ex_ids in zip(frames, frames_labels, frames_ids):
    for frame, label, frame_id in zip(ex_frames, ex_labels, ex_ids):
      savemat(directory +'/'+ str(counter+filename_offset) +'.mat', {'id':frame_id+id_offset, 'data': frame, 'label': label})
      counter += 1
  return counter+1, max(frames_ids)[0]+1

if __name__ == '__main__':

    dirname = os.path.dirname(__file__)
    data_dir = "" ## SET RAW DATA DIRECTORY HERE
    save_folder = ""
    folders = os.listdir(data_dir)

    columns_sheet = data_dir+'/FAME - Dataset Column Descriptions.xlsx'
    c_mat_files = os.listdir(data_dir+'/randomised C') # C-Control
    p_mat_files = os.listdir(data_dir+'/randomised P') # P-Pain 

    # Sorting to ensure always the same order
    c_mat_files.sort()
    p_mat_files.sort()

    # Loads .mat files
    p_mats, p_mat_files_filtered = load_mats(data_dir+'/randomised P/', p_mat_files)
    c_mats, c_mat_files_filtered = load_mats(data_dir+'/randomised C/', c_mat_files)

    # Exercise names
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

    # Generates lits of labelled data frames
    p_frames, p_frames_labels, p_frames_ids = process_data(p_mats, p_mat_files_filtered, columns_names, exercise_names)
    c_frames, c_frames_labels, c_frames_ids = process_data(c_mats, c_mat_files_filtered , columns_names, exercise_names)

    c_offset, id_offset = save_mats(p_frames , p_frames_labels, p_frames_ids, save_folder)
    save_mats(c_frames, c_frames_labels, c_frames_ids, save_folder, c_offset, id_offset)