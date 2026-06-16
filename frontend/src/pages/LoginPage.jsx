import { Heart, ShieldCheck } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useGoogleLogin } from "@react-oauth/google";

export default function LoginPage() {
const navigate = useNavigate();

const login = useGoogleLogin({
    onSuccess: async (tokenResponse) => {
        try {
            const res = await fetch(
                `https://www.googleapis.com/oauth2/v3/userinfo`,
                {
                    headers: {
                        Authorization: `Bearer ${tokenResponse.access_token}`,
                    },
                }
            );

            const user = await res.json();

            localStorage.setItem("user_email", user.email);
            localStorage.setItem("user_name", user.name);
            localStorage.setItem("user_picture", user.picture || "");

            navigate("/onboarding");
        } catch (err) {
            console.error(err);
        }
    },
});

return (
    <div className="grain min-h-screen bg-background text-foreground flex items-center justify-center px-6">
        <div className="w-full max-w-md glass-card p-10">
            <button
                onClick={() => navigate("/")}
                className="flex items-center gap-2 mb-8"
            >
                <div className="h-9 w-9 rounded-2xl bg-primary text-primary-foreground flex items-center justify-center">
                    <Heart size={18} />
                </div>

                <p
                    className="text-base font-semibold tracking-tight"
                    style={{ fontFamily: "Outfit" }}
                >
                    MediGuard AI
                </p>
            </button>

            <h1
                className="text-3xl md:text-4xl"
                style={{ fontFamily: "Outfit" }}
            >
                Welcome back.
            </h1>

            <p className="text-sm text-muted-foreground mt-3">
                Sign in to continue protecting your prescriptions.
            </p>

            <button
                onClick={() => login()}
                className="mt-10 w-full h-12 rounded-full bg-foreground text-background font-medium flex items-center justify-center gap-3 hover:opacity-90"
            >
                Continue with Google
            </button>

            <div className="mt-8 flex items-center gap-2 text-xs text-muted-foreground">
                <ShieldCheck size={14} />
                <p>HTTPS · session-based auth · revoke any time</p>
            </div>
        </div>
    </div>
);

}
