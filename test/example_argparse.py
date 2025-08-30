import argparse

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Example script with argparse")

    # Add arguments
    parser.add_argument("name", type=str, help="Your name")  # positional argument
    parser.add_argument("-a", "--age", type=int, default=18, help="Your age (default: 18)")  # optional argument
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")  # flag

    # Parse the arguments
    args = parser.parse_args()

    # Use them
    print(f"Hello, {args.name}!")
    print(f"Your age is {args.age}.")

    if args.verbose:
        print("Verbose mode is ON.")

if __name__ == "__main__":
    main()
