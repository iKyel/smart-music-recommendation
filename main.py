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

    # Define response model


# Load similarity matrix khi khởi động server
artist_similarity = None


def load_artist_similarity_matrix():
    """Load ma trận similarity từ file đã lưu

    Returns:
        DataFrame: Ma trận similarity giữa các playlist
    """
    try:
        # Load ma trận numpy
        similarity_values = np.load(
            'data/artist_recommend/artist_similarity_matrix.npy')

        # Load thông tin index từ file CSV
        index_df = pd.read_csv(
            'data/artist_recommend/artist_similarity_matrix.csv', index_col=0)

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
    top_similar = similar_scores.sort_values(
        ascending=False)[1:n_recommendations+1]

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
        artist_similarity = load_artist_similarity_matrix()
        if artist_similarity is None:
            raise HTTPException(
                status_code=500, detail="Could not load similarity matrix")

    # Kiểm tra artist_id có tồn tại
    if artist_id not in artist_similarity.index:
        raise HTTPException(
            status_code=404, detail=f"Artist ID {artist_id} not found")

    # Lấy đề xuất
    recommendations = recommend_artists(artist_id, artist_similarity, n)

    return ArtistRecommendationResponse(
        artist_id=artist_id,
        recommendations=[
            ArtistRecommendation(artist_id=aid, similarity_score=float(score))
            for aid, score in recommendations.items()
        ]
    )
class RecommendationResponse(BaseModel):
    recommendations: Dict[str, Dict[str, str]] 

# Khởi tạo ma trận và similarity matrix

def load_matrix():
    matrix_values = np.load(
        'data/track_recommend/playlist_song_matrix.npy')
    matrix_df = pd.read_csv(
        'data/track_recommend//playlist_song_matrix.csv', index_col=0)
    return pd.DataFrame(matrix_values, index=matrix_df.index, columns=matrix_df.columns)

def calculate_cosine_similarity(matrix):
    epsilon = 1e-10
    normalized = matrix.div(np.sqrt((matrix**2).sum(axis=1) + epsilon), axis=0)
    similarity = normalized.dot(normalized.T)
    return similarity


# Khởi tạo ma trận khi start server
matrix = load_matrix()
playlist_similarity = calculate_cosine_similarity(matrix)


@app.get("/recommend/tracks/{playlist_id}", response_model=List[Dict[str, str]])
async def recommend_tracks(playlist_id: str, n_recommendations: int = 5):
    """Đề xuất bài hát cho playlist

    Args:
        playlist_id: ID của playlist cần đề xuất
        n_recommendations: Số lượng bài hát muốn đề xuất (mặc định: 5)

    Returns:
        Dict[str, float]: Dictionary chứa track_id và điểm dự đoán
    """
    if playlist_id not in playlist_similarity.index:
        raise HTTPException(status_code=404, detail="Playlist không tồn tại")

    # Lấy top K playlist tương tự nhất
    similar_playlists = playlist_similarity.loc[playlist_id].sort_values(ascending=False)[
        1:11]

    # Dictionary để lưu điểm của các bài hát
    track_scores = {}

    # Lấy các bài hát hiện có trong playlist
    current_tracks = set(matrix.loc[playlist_id]
                         [matrix.loc[playlist_id] == 1].index)

    # Tính toán điểm cho các bài hát mới
    for similar_playlist_id, similarity_score in similar_playlists.items():
        similar_playlist_tracks = set(
            matrix.loc[similar_playlist_id][matrix.loc[similar_playlist_id] == 1].index
        )
        
        new_tracks = similar_playlist_tracks - current_tracks

        for track in new_tracks:
            if track not in track_scores:
                track_scores[track] = 0
            track_scores[track] += similarity_score

    if not track_scores:
        return {"recommendations": {}}

    # Sắp xếp và lấy top N bài hát
    sorted_tracks = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)[
        :n_recommendations]

    # Return only spotifyTrackId as a list of dictionaries
    normalized_recommendations = [
        {"spotifyTrackId": track}
        for track, _ in sorted_tracks
    ]

    return normalized_recommendations


# Define response model for track recommendations
class TrackRecommendation(BaseModel):
    track_id: str
    similarity_score: float

class TrackRecommendationResponse(BaseModel):
    track_id: str
    recommendations: List[TrackRecommendation]

# Initialize matrices when server starts
track_similarity = None
playlist_song_matrix = None

# Thêm hàm mới để load ma trận tương đồng của track từ file
def load_track_similarity_matrix():
    """Load ma trận tương đồng giữa các track từ file đã lưu
    
    Returns:
        DataFrame: Ma trận tương đồng giữa các track
    """
    try:
        # Load ma trận numpy
        similarity_values = np.load('data/track_recommend/track_similarity_matrix.npy')
        
        # Load thông tin index từ file CSV
        index_df = pd.read_csv('data/track_recommend/track_similarity_index.csv', header=None)
        track_ids = index_df[0].values
        
        # Tạo DataFrame với index đúng
        similarity_matrix = pd.DataFrame(
            similarity_values,
            index=track_ids,
            columns=track_ids
        )
        
        return similarity_matrix
    except Exception as e:
        print(f"Error loading track similarity matrix: {e}")
        return None

# Giữ nguyên hàm load_playlist_song_matrix() và recommend_tracks_by_track_id()
def load_playlist_song_matrix():
    """Load ma trận tương tác playlist-song từ file
    
    Returns:
        DataFrame: Ma trận tương tác playlist-song
    """
    try:
        # Load ma trận numpy
        matrix_values = np.load('data/track_recommend/playlist_song_matrix.npy')
        
        # Load thông tin index và columns từ file CSV
        matrix_df = pd.read_csv('data/track_recommend/playlist_song_matrix.csv', index_col=0)
        
        # Tạo DataFrame với index và columns đúng
        matrix = pd.DataFrame(
            matrix_values,
            index=matrix_df.index,
            columns=matrix_df.columns
        )
        
        return matrix
    except Exception as e:
        print(f"Error loading playlist-song matrix: {e}")
        return None

def calculate_track_similarity(matrix):
    """Tính toán độ tương đồng cosine giữa các bài hát

    Args:
        matrix (DataFrame): Ma trận tương tác playlist-song

    Returns:
        DataFrame: Ma trận độ tương đồng cosine giữa các bài hát
    """
    # Chuyển vị ma trận để có track làm hàng và playlist làm cột
    track_matrix = matrix.T
    
    # Chuẩn hóa ma trận theo hàng (track)
    # Thêm epsilon để tránh chia cho 0
    epsilon = 1e-10
    normalized = track_matrix.div(np.sqrt((track_matrix**2).sum(axis=1) + epsilon), axis=0)
    similarity = normalized.dot(normalized.T)

    return similarity

def recommend_tracks_by_track_id(track_id: str, track_similarity: pd.DataFrame, n_recommendations: int = 5) -> Dict[str, float]:
    """Đề xuất bài hát dựa trên track_id đầu vào

    Args:
        track_id (str): ID của bài hát cần tìm bài hát tương tự
        track_similarity (DataFrame): Ma trận độ tương đồng giữa các bài hát
        n_recommendations (int): Số lượng bài hát đề xuất

    Returns:
        Dict[str, float]: Dictionary chứa track_id và điểm tương đồng
    """
    if track_id not in track_similarity.index:
        return {}

    # Lấy độ tương đồng của track_id với tất cả các track khác
    similarities = track_similarity.loc[track_id].sort_values(ascending=False)
    
    # Loại bỏ chính track_id đó (similarity = 1.0)
    similarities = similarities[similarities.index != track_id]
    
    # Lấy top N bài hát tương tự nhất
    top_similar_tracks = similarities.head(n_recommendations)
    
    # Chuyển đổi thành dictionary với điểm được làm tròn
    return {track: round(float(score), 3) for track, score in top_similar_tracks.items()}

@app.get("/recommend/tracks/similar/{track_id}", response_model=TrackRecommendationResponse)
def get_similar_tracks(track_id: str, n: int = 5) -> TrackRecommendationResponse:
    """Endpoint để lấy danh sách bài hát tương tự dựa trên track_id

    Args:
        track_id (str): ID của bài hát cần tìm bài hát tương tự
        n (int, optional): Số lượng đề xuất. Defaults to 5.

    Returns:
        TrackRecommendationResponse: Danh sách các bài hát được đề xuất và điểm số
    """
    global track_similarity

    # Load ma trận tương đồng nếu chưa có
    if track_similarity is None:
        track_similarity = load_track_similarity_matrix()
        if track_similarity is None:
            raise HTTPException(status_code=500, detail="Could not load track similarity matrix")

    # Kiểm tra track_id có tồn tại
    if track_id not in track_similarity.index:
        raise HTTPException(status_code=404, detail=f"Track ID {track_id} not found")

    # Lấy đề xuất
    recommendations = recommend_tracks_by_track_id(track_id, track_similarity, n)

    return TrackRecommendationResponse(
        track_id=track_id,
        recommendations=[
            TrackRecommendation(track_id=tid, similarity_score=score)
            for tid, score in recommendations.items()
        ]
    )

if __name__ == "__main__":
    import uvicorn
    artist_similarity = load_artist_similarity_matrix()
    track_similarity = load_track_similarity_matrix()
    uvicorn.run(app, host=settings.host, port=settings.port)
