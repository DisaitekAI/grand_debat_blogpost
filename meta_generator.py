from nlp_demo_animation_function import generate_csv_file

def main():
    to_generate = {
        'fiscalite' : [
            11, 12, 13, 16, 17, 18
        ],
        'democratie': [
            11, 14, 17, 20, 23, 26, 27, 30, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 43, 44,
            45, 46, 47
        ],
        'transition': [
            12, 14, 16, 17, 18, 20, 22, 25, 26
        ],
        'organisation': [

        ],
    }

    for domain, question_indices in to_generate.items():
        for question_id in question_indices:
            generate_csv_file(domain, question_id)

if __name__ == '__main__':
    main()
