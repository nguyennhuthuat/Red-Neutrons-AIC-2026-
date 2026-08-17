# RED-NEUTRONS AIC retrieval UI

## Retrieval backends

The UI supports the existing CLIP index and an optional BLIP retrieval index.
The two models use different embedding spaces, so their vectors and FAISS
indexes are not interchangeable.

Build the existing CLIP artifacts first:

```powershell
python src/prepare_data.py
```
## TRAKE temporal action retrieval

Run the metadata-aware UI and select **TRAKE** for questions that request an
ordered sequence of moments from one video:

```powershell
streamlit run ui/enriched_search_ui.py
```

Gemini isolates the overall activity and produces an ordered linked list of
visible actions. The overall activity retrieves candidate videos; TRAKE then
scores every action against each candidate's full keyframe timeline and returns
up to three monotonic hypotheses for the top 20 videos.
