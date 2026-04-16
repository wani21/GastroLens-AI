import { NavLink } from "react-router-dom";

export default function Navbar() {
  const linkClass = ({ isActive }) =>
    `px-4 py-2 text-sm font-medium transition-colors ${
      isActive
        ? "text-accent border-b-2 border-accent"
        : "text-gray-600 hover:text-black"
    }`;

  return (
    <nav className="border-b border-gray-200 bg-white sticky top-0 z-10">
      <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
        <NavLink to="/" className="flex items-center gap-2">
          <div className="w-8 h-8 bg-black rounded flex items-center justify-center">
            <span className="text-accent font-bold text-lg">G</span>
          </div>
          <span className="font-bold text-lg tracking-tight">
            GastroLens<span className="text-accent">-AI</span>
          </span>
        </NavLink>
        <div className="flex items-center gap-2">
          <NavLink to="/" className={linkClass} end>
            Home
          </NavLink>
          <NavLink to="/image" className={linkClass}>
            Image
          </NavLink>
          <NavLink to="/video" className={linkClass}>
            Video
          </NavLink>
        </div>
      </div>
    </nav>
  );
}
