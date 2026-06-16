import { createContext, useContext, useEffect, useState, useCallback } from "react";
import { api } from "@/lib/api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    const refresh = useCallback(async () => {
        try {
            const email = localStorage.getItem("user_email");
            const name = localStorage.getItem("user_name");

            if (!email) {
                setUser(null);
                setLoading(false);
                return;
            }

            const res = await api.get("/auth/me", {
                headers: {
                    "X-User-Email": email,
                    "X-User-Name": name || "",
                },
            });

            setUser(res.data);
        } catch (err) {
            console.error(err);
            setUser(null);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        refresh();
    }, [refresh]);

    const logout = async () => {
        try {
            await api.post("/auth/logout");
        } catch (err) {
            console.error(err);
        }

        localStorage.removeItem("user_email");
        localStorage.removeItem("user_name");
        localStorage.removeItem("user_picture");

        setUser(null);
    };

    return (
        <AuthContext.Provider
            value={{
                user,
                setUser,
                loading,
                refresh,
                logout,
            }}
        >
            {children}
        </AuthContext.Provider>
    );
}

export const useAuth = () => useContext(AuthContext);