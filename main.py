from fastapi import FastAPI, HTTPException
from typing import Dict, List
from pydantic import BaseModel
from config import settings
import numpy as np
import pandas as pd

# Create FastAPI instance
app = FastAPI(
    title=settings.app_name,
    debug=settings.debug
)

# Define response model
class ArtistRecommendation(BaseModel):
    artist_id: str
    similarity_score: float

class ArtistRecommendationResponse(BaseModel):
    artist_id: str
    recommendations: List[ArtistRecommendation]

# Define response models for track recommendation
class TrackRecommendation(BaseModel):
    track_id: str
    similarity_score: float

class TrackRecommendationResponse(BaseModel):
    track_id: str
    recommendations: List[TrackRecommendation]

# Load similarity matrix khi khởi động server
artist_similarity = None

def load_similarity_matrix():
    """Load ma trận similarity từ file đã lưu

    Returns:
        DataFrame: Ma trận similarity giữa các playlist
    """
    try:
        # Load ma trận numpy
        similarity_values = np.load('data/artist_recommend/artist_similarity_matrix.npy')

        # Load thông tin index từ file CSV
        index_df = pd.read_csv('data/artist_recommend/artist_similarity_matrix.csv', index_col=0)

        # Tạo DataFrame với index đúng
        similarity_matrix = pd.DataFrame(
            similarity_values,
            index=index_df.index,
            columns=index_df.index
        )

        return similarity_matrix
    except Exception as e:
        print(f"Error loading similarity matrix: {e}")
        return None

# Load similarity matrix cho track khi khởi động server
track_similarity = None

def load_track_similarity_matrix():
    """Load ma trận similarity của track từ file đã lưu

    Returns:
        DataFrame: Ma trận similarity giữa các track
    """
    try:
        # Load ma trận numpy
        similarity_values = np.load('data/track_recommend/track_similarity_matrix.npy')

        # Load thông tin index từ file CSV
        index_df = pd.read_csv('data/track_recommend/track_similarity_index.csv', index_col=0)

        # Tạo DataFrame với index đúng
        similarity_matrix = pd.DataFrame(
            similarity_values,
            index=index_df.index,
            columns=index_df.index
        )

        return similarity_matrix
    except Exception as e:
        print(f"Error loading track similarity matrix: {e}")
        return None

def recommend_artists(artist_id: str, similarity_matrix: pd.DataFrame, n_recommendations: int = 5) -> Dict[str, float]:
    """Đề xuất các nghệ sĩ tương tự dựa trên artist_id

    Args:
        artist_id (str): ID của nghệ sĩ cần tìm đề xuất
        similarity_matrix (DataFrame): Ma trận similarity
        n_recommendations (int, optional): Số lượng đề xuất. Defaults to 5.

    Returns:
        Dict[str, float]: Dictionary chứa artist_id và điểm số similarity
    """
    if artist_id not in similarity_matrix.index:
        return {}

    # Lấy hàng tương ứng với artist_id
    similar_scores = similarity_matrix.loc[artist_id]

    # Sắp xếp và lấy top n_recommendations
    top_similar = similar_scores.sort_values(ascending=False)[1:n_recommendations+1]

    return dict(top_similar)

@app.get("/recommend/artists/{artist_id}", response_model=ArtistRecommendationResponse)
def get_similar_artists(artist_id: str, n: int = 5) -> ArtistRecommendationResponse:
    """Endpoint để lấy danh sách nghệ sĩ tương tự

    Args:
        artist_id (str): ID của nghệ sĩ cần tìm đề xuất
        n (int, optional): Số lượng đề xuất. Defaults to 5.

    Returns:
        ArtistRecommendationResponse: Danh sách các nghệ sĩ được đề xuất và điểm số
    """
    global artist_similarity

    # Load similarity matrix nếu chưa được load
    if artist_similarity is None:
        artist_similarity = load_similarity_matrix()
        if artist_similarity is None:
            raise HTTPException(status_code=500, detail="Could not load similarity matrix")

    # Kiểm tra artist_id có tồn tại
    if artist_id not in artist_similarity.index:
        raise HTTPException(status_code=404, detail=f"Artist ID {artist_id} not found")

    # Lấy đề xuất
    recommendations = recommend_artists(artist_id, artist_similarity, n)

    return ArtistRecommendationResponse(
        artist_id=artist_id,
        recommendations=[
            ArtistRecommendation(artist_id=aid, similarity_score=float(score))
            for aid, score in recommendations.items()
        ]
    )

# Root endpoint và health check endpoint giữ nguyên

@app.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "Welcome to Smart Music Recommendation System"}

def recommend_tracks(track_id: str, similarity_matrix: pd.DataFrame, n_recommendations: int = 5) -> Dict[str, float]:
    """Đề xuất các track tương tự dựa trên track_id

    Args:
        track_id (str): ID của track cần tìm đề xuất
        similarity_matrix (DataFrame): Ma trận similarity
        n_recommendations (int, optional): Số lượng đề xuất. Defaults to 5.

    Returns:
        Dict[str, float]: Dictionary chứa track_id và điểm số similarity
    """
    if track_id not in similarity_matrix.index:
        return {}

    # Lấy hàng tương ứng với track_id
    similar_scores = similarity_matrix.loc[track_id]

    # Sắp xếp và lấy top n_recommendations
    top_similar = similar_scores.sort_values(ascending=False)[1:n_recommendations+1]

    return dict(top_similar)

# Endpoint để lấy danh sách track tương tự
@app.get("/recommend/tracks/{track_id}", response_model=TrackRecommendationResponse)
def get_similar_tracks(track_id: str, n: int = 5) -> TrackRecommendationResponse:
    """Endpoint để lấy danh sách track tương tự

    Args:
        track_id (str): ID của track cần tìm đề xuất
        n (int, optional): Số lượng đề xuất. Defaults to 5.

    Returns:
        TrackRecommendationResponse: Danh sách các track được đề xuất và điểm số
    """
    global track_similarity

    # Load similarity matrix nếu chưa được load
    if track_similarity is None:
        track_similarity = load_track_similarity_matrix()
        if track_similarity is None:
            raise HTTPException(status_code=500, detail="Could not load track similarity matrix")

    # Kiểm tra track_id có tồn tại
    if track_id not in track_similarity.index:
        raise HTTPException(status_code=404, detail=f"Track ID {track_id} not found")

    # Lấy đề xuất
    recommendations = recommend_tracks(track_id, track_similarity, n)

    return TrackRecommendationResponse(
        track_id=track_id,
        recommendations=[
            TrackRecommendation(track_id=tid, similarity_score=float(score))
            for tid, score in recommendations.items()
        ]
    )

if __name__ == "__main__":
    import uvicorn
    # Load similarity matrix khi khởi động
    artist_similarity = load_similarity_matrix()
    track_similarity = load_track_similarity_matrix()
    uvicorn.run(app, host=settings.host, port=settings.port)
