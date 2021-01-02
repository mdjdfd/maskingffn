import masking as rm


def main():
    masked_tensor = rm.random_masking_first_layer()
    print(masked_tensor)


if __name__ == "__main__":
    main()
