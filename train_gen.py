import random
import argparse


def generate_samples(start, end, num_samples, output_file):
    # Generate unique random numbers
    population = list(range(start, end + 1))
    if num_samples > len(population):
        raise ValueError("Sample size cannot exceed population size")

    samples = random.sample(population, num_samples)

    # Write to file with 6-digit formatting
    with open(output_file, 'w') as f:
        for number in samples:
            f.write(f"{number:06d}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random unique numbers in a range')
    parser.add_argument('--start', type=int, required=True, help='Start of range')
    parser.add_argument('--end', type=int, required=True, help='End of range (inclusive)')
    parser.add_argument('--samples', type=int, required=True, help='Number of samples to generate')
    parser.add_argument('--output', type=str, required=True, help='Output file name')

    args = parser.parse_args()

    try:
        generate_samples(args.start, args.end, args.samples, args.output)
        print(f"Successfully generated {args.samples} unique numbers in {args.output}")
    except ValueError as e:
        print(f"Error: {str(e)}")
