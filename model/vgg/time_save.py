import csv

def time_save(start_time, end_time, filename, model_name):

  time_taken = (end_time - start_time) / 60
  print(f'걸린 시간: {time_taken} 분')

  try:
      with open(filename, mode='x', newline='') as file:
          writer = csv.writer(file)
          writer.writerow(['Model', 'Training Time'])

  except FileExistsError:
      with open(filename, mode='a', newline='') as file:
          writer = csv.writer(file)
          writer.writerow([model_name, time_taken])