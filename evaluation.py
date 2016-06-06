import numpy as np
# Read ratings

users = {}
i = 0
with open("/home/alfredo/Desktop/bpr/kaggle_users.txt") as f:
    for line in f:
        users[line.strip()] = i
        i += 1

songs = {}
with open("/home/alfredo/Desktop/bpr/kaggle_songs.txt") as f:
    for line in f:
        song, id = line.strip().split(" ")
        songs[song] = int(id)

ratings = []
with open("/home/alfredo/Desktop/bpr/kaggle_visible_evaluation_triplets.txt") as f:
    for line in f:
        user, song, _ = line.strip().split("\t")
        ratings.append((users[user], songs[song]))

# Each user at least 10 songs, each song at least 10 users
songs_by_user = {}
users_by_song = {}
for rating in ratings:
    if rating[0] in songs_by_user:
        songs_by_user[rating[0]].append(rating[1])
    else:
        songs_by_user[rating[0]] = [rating[1]]

    if rating[1] in users_by_song:
        users_by_song[rating[1]].append(rating[0])
    else:
        users_by_song[rating[1]] = [rating[0]]

surviving_users = {}
for user in songs_by_user:
    if len(songs_by_user[user]) >= 10:
        surviving_users[user] = True

surviving_songs = {}
for song in users_by_song:
    if len(users_by_song[song]) >= 10:
        surviving_songs[song] = True

final_ratings = []
for rating in ratings:
    if rating[0] in surviving_users and rating[1] in surviving_songs:
        final_ratings.append(rating)

final_ratings_by_user = {}
for rating in final_ratings:
    if rating[0] in final_ratings_by_user:
        final_ratings_by_user[rating[0]].append(rating[1])
    else:
        final_ratings_by_user[rating[0]] = [rating[1]]

train_ratings = []
test_ratings = []
test_rating_by_user = {}
for user in final_ratings_by_user:
    # First element goes to test set, rest to training
    test_ratings.append((user, final_ratings_by_user[user][0]))
    test_rating_by_user[user] = final_ratings_by_user[user][0]

    for song in final_ratings_by_user[user][1:]:
        train_ratings.append((user, song))

# Write training elements in file
with open("/home/alfredo/Desktop/bpr/training_ratings.txt", "w") as f:
    for rating in train_ratings:
        f.write("%d %d\n" % (rating[0], rating[1]))



############################
### EVALUATION
############################
# Read model matrices
userMat = np.loadtxt("/home/alfredo/Desktop/bpr/userMatrix.txt")
prodMat = np.loadtxt("/home/alfredo/Desktop/bpr/prodMatrix.txt")

songs = {}
for rating in final_ratings:
    if rating[1] not in songs:
        songs[rating[1]] = True

# Compute AUC
total_auc = 0
total_users = 0
for user in final_ratings_by_user.keys()[:1000]: # Choose a sample of users (all is too much)
    # Only one test rating by user, as seen in the BPR paper
    song_test = test_rating_by_user[user]
    num_ratings = 0
    auc_user = 0
    for song in songs.keys():
        if song not in final_ratings_by_user[user]:
            testScore = userMat[user-1].dot(prodMat[song_test])
            otherScore = userMat[user-1].dot(prodMat[song])

            # We want the test song score to be higher
            if testScore > otherScore:
                auc_user += 1
            num_ratings += 1

    auc_user = auc_user * 1.0 / num_ratings
    total_auc += auc_user
    total_users += 1
    print ("User: %d, AUC: %f" % (total_users, auc_user))

total_auc = total_auc / len(final_ratings_by_user.keys()[:1000])
print ("Total AUC is %f" % total_auc)