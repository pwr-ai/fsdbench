from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server import BenchmarkServer


def create_app(server: BenchmarkServer):
    """Create a FastAPI application wrapping the benchmark server."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:
        raise ImportError(
            "FastAPI is required for the HTTP server. "
            "Install with: pip install 'fsdbench[serve]'"
        ) from exc

    app = FastAPI(
        title="Factual Discovery Benchmark",
        description="Benchmark server for testing factual-state discovery methods",
    )

    class LoadSampleRequest(BaseModel):
        sample_idx: int

    class AskRequest(BaseModel):
        question: str

    class AskResponse(BaseModel):
        answer: str
        questions_asked: int
        answers_collected: int

    @app.get("/")
    def root():
        return {"status": "ok", "info": server.info()}

    @app.get("/info")
    def get_info():
        return server.info()

    @app.post("/load_sample")
    def load_sample(request: LoadSampleRequest):
        try:
            factual_state = server.load_sample(request.sample_idx)
            return {
                "status": "loaded",
                "sample_idx": request.sample_idx,
                "factual_state_length": len(factual_state),
            }
        except IndexError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.get("/factual_state")
    def get_factual_state():
        if server.factual_state is None:
            raise HTTPException(status_code=400, detail="No sample loaded")
        return {"factual_state": server.factual_state}

    @app.post("/ask")
    def ask(request: AskRequest) -> AskResponse:
        if server.factual_state is None:
            raise HTTPException(status_code=400, detail="No sample loaded")
        answer = server.ask(request.question)
        server_info = server.info()
        return AskResponse(
            answer=answer,
            questions_asked=server_info["questions_asked"],
            answers_collected=server_info["answers_collected"],
        )

    @app.get("/score")
    def get_score():
        if server.factual_state is None:
            raise HTTPException(status_code=400, detail="No sample loaded")
        return server.score()

    @app.post("/reset")
    def reset():
        if server.factual_state is None:
            raise HTTPException(status_code=400, detail="No sample loaded")
        server.reset()
        return {"status": "reset", "info": server.info()}

    @app.get("/answers")
    def get_answers():
        return {"answers": server.get_answers()}

    @app.get("/history")
    def get_history():
        return {"history": server.get_history()}

    @app.post("/flush_log")
    def flush_log():
        path = server.flush_log()
        if path is None:
            return {"status": "logging_disabled"}
        return {"status": "saved", "path": str(path)}

    return app


def serve(
    model: str = "gpt-4o-mini",
    port: int = 8000,
    dataset_path: str | None = None,
    log_dir: str | None = None,
    log_judge_calls: bool = False,
    run_name: str | None = None,
) -> None:
    """Start the benchmark HTTP server."""
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn is required to serve. "
            "Install with: pip install 'fsdbench[serve]'"
        ) from exc

    from .server import BenchmarkServer

    server_kwargs = dict(
        model=model, verbose=True,
        log_dir=log_dir, log_judge_calls=log_judge_calls,
        run_name=run_name,
    )
    if dataset_path:
        server_kwargs["dataset_path"] = dataset_path
    server = BenchmarkServer(**server_kwargs)
    app = create_app(server)
    uvicorn.run(app, host="0.0.0.0", port=port)
