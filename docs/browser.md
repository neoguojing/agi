           ┌──────────────┐
           │   Agent      │
           └──────┬───────┘
                  │
        ┌─────────▼─────────┐
        │ BrowserMiddleware │
        └─────────┬─────────┘
                  │
   ┌──────────────┼──────────────┐
   │              │              │
BrowserBackend   OCR Service   Storage
(Playwright)     (LLM/OCR)     (FS/Vector)