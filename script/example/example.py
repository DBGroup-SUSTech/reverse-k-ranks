import numpy as np
import gen_example


def normalization(user_l):
    norm_user_l = user_l.copy()
    for i in range(len(user_l)):
        norm = np.linalg.norm(user_l[i])
        norm_user_l[i] = norm_user_l[i] / norm
    return norm_user_l


def user_item_norm():
    item_l = np.array(
        [[0.7585935, 1.6275389],
         [1.7348094, 2.3759346],
         [3.1377952, 0.08928549],
         [1.8950846, 3.26434],
         [3.1293406, 1.0796206]], dtype=np.float64
    )

    user_l = np.array(
        [[0.13834432, 2.2017198],
         [2.9890387, 2.9058983],
         [0.6941855, 0.22097017],
         [1.6679517, 1.9731998]],
        dtype=np.float64)

    IP_l = np.dot(user_l, item_l.T)
    before_ip_l = np.array([np.sort(-_) for _ in IP_l])
    before_ip_l = -before_ip_l
    print("before")
    print(before_ip_l.T)
    norm_user_l = normalization(user_l)
    IP_l = np.dot(norm_user_l, item_l.T)
    after_ip_l = np.array([np.sort(-_) for _ in IP_l])
    after_ip_l = -after_ip_l
    print("after")
    print(after_ip_l.T)


def q_norm():
    q = np.array([0.0, 8.2, 8.8])
    norm_q = np.linalg.norm(q)
    print(norm_q)


def compute_rank(user, item, query):
    score_table = np.dot(user, item.T)  # 行是user * item_l
    queryIP_l = np.dot(query, user.T)
    query_rank_l = [(queryIP_l[0][userID] <= np.array(score_table[userID])).sum() + 1 for userID in range(len(user_l))]
    return query_rank_l, queryIP_l[0], score_table


if __name__ == '__main__':
    item_l = np.array(
        [[1.9, 0.5],
         [0.2, 2.7],
         [1.4, 0.5],
         [0.5, 1.9],
         [0.9, 0.9],
         [0.8, 1.8],
         [2.0, 0.]], dtype=np.float32
    )
    query_l = np.array([[1.7, 0.6]], dtype=np.float32)

    user_l = np.array(
        [[1.5, 0.9],
         [1.9, 1.8],
         [0.3, 2.7],
         [1.2, 0.],
         [1.5, 1.2]],
        dtype=np.float32)

    query_rank_l, queryIP_l, score_table = compute_rank(user_l, item_l, query_l)
    print(query_rank_l)
    print(queryIP_l)
    print(score_table)
    sort_list = np.array([np.sort(score_list) for score_list in score_table])
    print("sort_score_table\n", sort_list)
    arg_sort_list = np.array([np.argsort(score_list) + 1 for score_list in score_table])
    print("arg_sort_score_table\n", arg_sort_list)

    # gnd_id, gnd_dist = gen_example.ip_gnd(item_l, user_l, len(item_l))
    # gnd_id = gnd_id + 1
    # print(gnd_id)
    # print(gnd_dist)
