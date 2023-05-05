import json
from models.SocialMediaPost import SocialMediaPost

def get_social_media_posts(json_file):
    with open(json_file, 'r') as file:
        social_media_posts = []
        i =0
        for line in file:
            print(i)
            i = i+1
            data = json.loads(line)
            post = SocialMediaPost(
                text=data['text'],
                date=data['date'],
                label=data['label'],
                id=data['id'],
                label_name=data['label_name']
            )
            social_media_posts.append(post)
    print(social_media_posts)
    return social_media_posts