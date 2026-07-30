"""Assistant tools — non-RAG capabilities.

These tools read MongoDB directly and never use Qdrant or semantic search:
  - conversation summarization
  - conversation insights (DM list with activity summaries)
  - conversation lookup (a DM with one named person)
  - community / channel summarization (metadata + recent activity)
  - friend / community / creator recommendations
  - social graph listing (actual friends, communities, followed creators)
  - community lookup (membership check / specific community search)
  - social overview (aggregate stats / dashboard)
  - user search (global ACL-safe people search)
  - voice session recall (most recent summarized call in a channel)
"""

from app.tools.catch_me_up import get_catch_me_up_data
from app.tools.community_lookup import lookup_my_community
from app.tools.community_recommender import suggest_communities
from app.tools.community_summary import get_channel_summary, get_community_summary
from app.tools.conversation_insights import get_conversation_insights
from app.tools.conversation_lookup import lookup_conversation_with
from app.tools.conversation_summary import build_conversation_summary
from app.tools.creator_recommender import suggest_creators
from app.tools.friend_recommender import suggest_friends
from app.tools.list_communities import get_my_communities
from app.tools.list_creators import get_followed_creators
from app.tools.list_friends import get_my_friends
from app.tools.social_overview import get_social_overview
from app.tools.trending_creators import get_trending_creators
from app.tools.user_search import search_users
from app.tools.voice_session_recall import get_voice_session_recall

__all__ = [
    "build_conversation_summary",
    "get_catch_me_up_data",
    "get_channel_summary",
    "get_community_summary",
    "get_conversation_insights",
    "get_followed_creators",
    "get_my_communities",
    "get_my_friends",
    "get_social_overview",
    "get_trending_creators",
    "get_voice_session_recall",
    "lookup_conversation_with",
    "lookup_my_community",
    "search_users",
    "suggest_communities",
    "suggest_creators",
    "suggest_friends",
]
