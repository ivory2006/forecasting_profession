import json
from os.path import join, dirname
from watson_developer_cloud import PersonalityInsightsV3

"""
The example returns a JSON response whose content is the same as that in
   ../resources/personality-v3-expect2.txt
"""

personality_insights = PersonalityInsightsV3(
    version='2016-10-20',
    username='667f989a-04d6-41d1-8570-36a357b70038',
    password='QUOsz7VRq22A')

with open(join(dirname(__file__), 'ssb4.txt')) as profile_json:
    profile = personality_insights.profile(
        profile_json.read(), 
        content_type='text/plain',
        content_language='en',
        raw_scores=True, consumption_preferences=True)
    fname="ssb4.json"
    with open(fname,'a') as f:
        f.write(json.dumps(profile, indent=2))
        f.close()
   # print(json.dumps(profile, indent=2))

