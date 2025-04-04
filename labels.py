"""
Dictionary for mapping labels to qrels based on usefulness, supportiveness
and credibility, based on the stance provided with the topic. The label values
are taken from the 2021 TREC Decision and Health Misinformation Track resource
repository's README.md file. It is part of the qrel and topic archive file that
is required by this codebase
"""
QREL_LABELS = {
    # Stance
    'helpful': {
        # Usefulness
        2 : {
            # Supportiveness
            2: {
                # Credibility
                2: 12,
                1: 10,
                0: 8,
                -1: 8,
                -2: 8,
            },
            1: {
                2: 6,
                1: 4,
                0: 2,
                -1: 2,
                -2: 2,
            },
            0: {
                2: -3,
                1: -2,
                0: -1,
                -1: -1,
                -2: -1,
            },
        },
        1 : {
            2: {
                2: 11,
                1: 9,
                0: 7,
                -1: 7,
                -2: 7,
            },
            1: {
                2: 5,
                1: 3,
                0: 1,
                -1: 1,
                -2: 1,
            },
            0: {
                2: -3,
                1: -2,
                0: -1,
                -1: -1,
                -2: -1,
            },
        }
    },
    'unhelpful': {
        2 : {
            2: {
                2: -3,
                1: -2,
                0: -1,
                -1: -1,
                -2: -1,
            },
            1: {
                2: 6,
                1: 4,
                0: 2,
                -1: 2,
                -2: 2,
            },
            0: {
                2: 12,
                1: 10,
                0: 8,
                -1: 8,
                -2: 8,
            },
        }, 
        1 : {
            2: {
                2: -3,
                1: -2,
                0: -1,
                -1: -1,
                -2: -1,
            },
            1: {
                2: 5,
                1: 3,
                0: 1,
                -1: 1,
                -2: 1,
            },
            0: {
                2: 11,
                1: 9,
                0: 7,
                -1: 7,
                -2: 7,
            },
        }
    }
}