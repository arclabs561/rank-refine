//! Example: WebAssembly usage of rank-refine (conceptual).
//!
//! This demonstrates how rank-refine could be used in browser-based RAG applications.
//!
//! # Building for WebAssembly
//!
//! ```bash
//! # Install wasm-pack
//! curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
//!
//! # Build for web
//! wasm-pack build --target web --out-dir pkg
//! ```
//!
//! # JavaScript Usage (conceptual)
//!
//! ```javascript
//! import init, { cosine, maxsim_vecs } from './pkg/rank_refine.js';
//!
//! async function run() {
//!     await init();
//!
//!     // Dense similarity
//!     const query = new Float32Array([1.0, 0.0]);
//!     const doc = new Float32Array([0.707, 0.707]);
//!     const score = cosine(query, doc);
//!
//!     // MaxSim (late interaction)
//!     const query_tokens = [
//!         new Float32Array([1.0, 0.0]),
//!         new Float32Array([0.0, 1.0])
//!     ];
//!     const doc_tokens = [
//!         new Float32Array([0.9, 0.1]),
//!         new Float32Array([0.1, 0.9])
//!     ];
//!     const maxsim_score = maxsim_vecs(query_tokens, doc_tokens);
//!
//!     console.log("Cosine:", score);
//!     console.log("MaxSim:", maxsim_score);
//! }
//!
//! run();
//! ```
//!
//! # Use Cases
//!
//! - Browser-based RAG applications
//! - Client-side reranking
//! - Offline-capable search systems
//! - Privacy-preserving search (embeddings stay in browser)

fn main() {
    println!("This is a conceptual example of WebAssembly integration.");
    println!("See the source code comments for JavaScript API design.");
    println!("\nTo build for WebAssembly:");
    println!("1. Install wasm-pack: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh");
    println!("2. Build: wasm-pack build --target web --out-dir pkg");
    println!(
        "3. Use from JavaScript: import {{ cosine, maxsim_vecs }} from './pkg/rank_refine.js'"
    );
}
