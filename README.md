# RED-NEUTRONS AIC retrieval UI

TRAKE Implementation

## HOW TO USE 


### 1. Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Add retrieval artifacts

Create `data/processed/` and provide one complete artifact set.

Artifacts - features npy file, metadata, faiss index - Lấy từ SigLIP2 của Thắng

### 3. Set configure cho GEMINI

Set API KEY trong PowerShell
```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
$env:GEMINI_TRAKE_MODEL="YOUR_GEMINI_MODEL"
$env:TRAKE_TIMEOUT_SECONDS="15"
```
## 4. TRAKE temporal action retrieval

Run the metadata-aware UI and select **TRAKE** for questions that request an
ordered sequence of moments from one video:

```powershell
streamlit run ui/enriched_search_ui.py
```

**Gemini** isolates the overall activity and produces an ordered linked list of
visible actions. The overall activity retrieves candidate videos; TRAKE then
scores every action against each candidate's full keyframe timeline and returns
up to three monotonic hypotheses for the top 20 videos.
