import random

split_ratio = 0.2  # 20% for validation, adjust as needed

total_numbers = 7481
split_index = int(total_numbers * split_ratio)

numbers = list(range(total_numbers))
random.shuffle(numbers)

train_numbers = numbers[split_index:]
val_numbers = numbers[:split_index]

# Format train numbers as zero-padded strings
train_numbers_formatted = [str(num).zfill(6) for num in train_numbers]

# Format validation numbers as zero-padded strings
val_numbers_formatted = [str(num).zfill(6) for num in val_numbers]


# Write train numbers to train.txt
with open("train.txt", "w") as train_file:
    train_file.write("\n".join(map(str, sorted(train_numbers_formatted))))

# Write validation numbers to val.txt
with open("val.txt", "w") as val_file:
    val_file.write("\n".join(map(str, sorted(val_numbers_formatted))))
