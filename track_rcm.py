import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

app = FastAPI(
    title="Playlist Song Recommendation API",
    description="API để đề xuất bài hát cho playlist dựa trên ma trận tương tác",
    version="1.0.0"
)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)