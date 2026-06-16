import { useEffect, useRef, useState } from "react";
import { api, API_BASE } from "@/lib/api";
import { motion } from "framer-motion";
import { Send, Mic, MicOff, Bot, User, Volume2, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";

export default function Copilot() {
    const { user } = useAuth();
    const language = user?.profile?.language || "en";
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [sessionId, setSessionId] = useState(null);
    const [streaming, setStreaming] = useState(false);
    const [recording, setRecording] = useState(false);
    const [sttBusy, setSttBusy] = useState(false);
    const [ttsPlaying, setTtsPlaying] = useState(null);
    const recorderRef = useRef(null);
    const chunksRef = useRef([]);
    const endRef = useRef(null);

    useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

    const send = async (text) => {
        const content = (text ?? input).trim();
        if (!content || streaming) return;
        setInput("");
        const userMsg = { role: "user", content, id: Math.random() };
        setMessages((m) => [...m, userMsg, { role: "assistant", content: "", id: "stream" }]);
        setStreaming(true);
        try {
            const res = await fetch(`${API_BASE}/copilot/chat`, {
    method: "POST",
    credentials: "include",
    headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${localStorage.getItem("user_email")}`,
    },
    body: JSON.stringify({
        message: content,
        chat_session_id: sessionId,
        language,
    }),
});
            if (!res.ok || !res.body) throw new Error("stream failed");
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let accumulated = "";
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop() || "";
                for (const line of lines) {
                    if (!line.startsWith("data:")) continue;
                    try {
                        const payload = JSON.parse(line.slice(5).trim());
                        if (payload.delta) {
                            accumulated += payload.delta;
                            setMessages((m) => m.map((x) => x.id === "stream" ? { ...x, content: accumulated } : x));
                        }
                        if (payload.done) {
                            setSessionId(payload.session_id);
                        }
                        if (payload.error) {
                            toast.error(payload.error);
                        }
                    } catch {}
                }
            }
            setMessages((m) => m.map((x) => x.id === "stream" ? { ...x, id: Math.random() } : x));
        } catch (e) {
            toast.error("Copilot stream failed");
        } finally {
            setStreaming(false);
        }
    };

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mime = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
            const rec = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
            chunksRef.current = [];
            rec.ondataavailable = (e) => { if (e.data.size) chunksRef.current.push(e.data); };
            rec.onstop = async () => {
                stream.getTracks().forEach((t) => t.stop());
                const blob = new Blob(chunksRef.current, { type: rec.mimeType || "audio/webm" });
                await uploadAudio(blob);
            };
            rec.start();
            recorderRef.current = rec;
            setRecording(true);
        } catch (e) {
            toast.error("Microphone unavailable");
        }
    };

    const stopRecording = () => {
        if (recorderRef.current && recorderRef.current.state !== "inactive") {
            recorderRef.current.stop();
        }
        setRecording(false);
    };

    const uploadAudio = async (blob) => {
        setSttBusy(true);
        try {
            const fd = new FormData();
            fd.append("file", blob, "rec.webm");
            if (language) fd.append("language", language);
            const r = await api.post("/voice/transcribe", fd, { headers: { "Content-Type": "multipart/form-data" } });
            const text = r.data?.text?.trim();
            if (text) await send(text);
        } catch (e) {
            toast.error("Voice transcription failed");
        } finally {
            setSttBusy(false);
        }
    };

    const playTts = async (idx, text) => {
        if (ttsPlaying === idx) return;
        setTtsPlaying(idx);
        try {
            const res = await fetch(`${API_BASE}/voice/tts`, {
                method: "POST",
                credentials: "include",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text }),
            });
            if (!res.ok) throw new Error();
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);
            audio.onended = () => { setTtsPlaying(null); URL.revokeObjectURL(url); };
            audio.onerror = () => { setTtsPlaying(null); };
            audio.play();
        } catch {
            toast.error("Voice playback failed");
            setTtsPlaying(null);
        }
    };

    const suggestions = [
        "Why am I taking these medicines?",
        "Can I take all of them together?",
        "What if I miss a dose?",
        "Side effects I should watch for?",
    ];

    return (
        <div className="space-y-6">
            <div>
                <p className="text-xs uppercase tracking-widest text-muted-foreground">Powered by Gemini 3 Pro</p>
                <h1 className="text-4xl md:text-5xl mt-1" style={{fontFamily:"Outfit"}}>AI Medical Copilot</h1>
            </div>

            <div className="glass-card p-6 h-[60vh] flex flex-col" data-testid="copilot-chat">
                <div className="flex-1 overflow-y-auto space-y-4 pr-2">
                    {messages.length === 0 && (
                        <div className="h-full flex flex-col items-center justify-center text-center">
                            <Bot size={28} className="text-primary"/>
                            <p className="mt-3 font-medium" style={{fontFamily:"Outfit"}}>Ask anything about your medicines</p>
                            <p className="text-xs text-muted-foreground mt-1">Responses use your prescription + profile context.</p>
                            <div className="mt-6 flex flex-wrap justify-center gap-2 max-w-xl">
                                {suggestions.map((s) => (
                                    <button key={s} onClick={() => send(s)} data-testid={`suggest-${s}`}
                                        className="px-3 py-2 rounded-full border border-border text-xs hover:bg-muted">{s}</button>
                                ))}
                            </div>
                        </div>
                    )}
                    {messages.map((m, idx) => (
                        <motion.div key={m.id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className={`flex gap-3 ${m.role === "user" ? "justify-end" : ""}`}>
                            {m.role === "assistant" && <div className="h-8 w-8 rounded-full bg-primary/15 text-primary flex items-center justify-center shrink-0"><Bot size={16}/></div>}
                            <div className={`max-w-[80%] p-3 rounded-2xl text-sm ${m.role === "user" ? "bg-primary text-primary-foreground rounded-br-sm" : "bg-muted rounded-bl-sm"}`}>
                                <p className="whitespace-pre-wrap leading-relaxed">{m.content || (m.id === "stream" ? "…" : "")}</p>
                                {m.role === "assistant" && m.content && (
                                    <button onClick={() => playTts(idx, m.content)} data-testid={`tts-${idx}`} className="mt-2 text-xs text-muted-foreground hover:text-foreground flex items-center gap-1">
                                        {ttsPlaying === idx ? <Loader2 size={12} className="animate-spin"/> : <Volume2 size={12}/>} Read aloud
                                    </button>
                                )}
                            </div>
                            {m.role === "user" && <div className="h-8 w-8 rounded-full bg-secondary/20 text-secondary flex items-center justify-center shrink-0"><User size={16}/></div>}
                        </motion.div>
                    ))}
                    <div ref={endRef}/>
                </div>
                <form onSubmit={(e) => { e.preventDefault(); send(); }} className="mt-4 flex items-center gap-2">
                    <button type="button" onClick={recording ? stopRecording : startRecording} data-testid="voice-rec-btn"
                        className={`h-12 w-12 rounded-full flex items-center justify-center ${recording ? "bg-emergency-red text-white animate-pulse" : "border border-border hover:bg-muted"}`}>
                        {sttBusy ? <Loader2 size={16} className="animate-spin"/> : recording ? <MicOff size={16}/> : <Mic size={16}/>}
                    </button>
                    <input data-testid="copilot-input" value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask about your medication…"
                        className="flex-1 h-12 px-4 rounded-full bg-background border border-border focus:outline-none focus:ring-2 focus:ring-primary/50"/>
                    <button type="submit" disabled={streaming || !input.trim()} data-testid="copilot-send-btn"
                        className="h-12 px-5 rounded-full bg-primary text-primary-foreground text-sm flex items-center gap-2 disabled:opacity-50">
                        <Send size={16}/>
                    </button>
                </form>
            </div>
        </div>
    );
}
