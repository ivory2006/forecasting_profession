import json
from os.path import join, dirname
from watson_developer_cloud import PersonalityInsightsV3

"""
The example returns a JSON response whose content is the same as that in
   profile.json
"""

personality_insights = PersonalityInsightsV3(
    version='2016-10-20',
    username='fb940cff-c2c7-4f75-85aa-bf3d80d51fbf',
    password='Be0iQXvS5KkW')

with open(join(dirname(__file__), 'profile.json')) as profile_json:
    profile = personality_insights.profile(
        profile_json.read(), 
        content_type='application/json',
        raw_scores=True, consumption_preferences=True)

    print(json.dumps(profile, indent=2))

