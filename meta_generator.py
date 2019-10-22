from nlp_demo_animation_function import generate_csv_file

def main():
    to_generate = {
        'fiscalite' : [
            11, 12, 13,
        ],
        'democratie': [

        ],
        'transition': [

        ],
        'organisation': [

        ],
    }

    for domain, question_indices in to_generate.items():
        for question_id in question_indices:
            generate_csv_file(domain, question_id)

if __name__ == '__main__':
    main()
