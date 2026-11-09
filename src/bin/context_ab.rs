//! Translates sentences whose correct form depends on earlier chat messages, once with that
//! context and once without, and checks the context-only forms (gender, formality, word sense).
//! Also checks context never leaks into the output.
//!
//! `--prompt N` swaps in an alternative context prompt (see `user_prompt`) to compare wordings.
//!
//! Usage: `context_ab <model_dir> [--mode nf4g128] [--threads N] [--cpu] [--prompt 0..3]`

use kronos_2t::runtime::chat::{ChatModel, Message};
use kronos_2t::runtime::translate::{clean_translation, prompt, run, TranslateRequest};
use std::path::PathBuf;

struct Case {
    to: &'static str,
    text: &'static str,
    context: &'static [&'static str],
    /// Any of these (lowercase) in the with-context output means the context was used.
    expect: &'static [&'static str],
    why: &'static str,
}

const CASES: &[Case] = &[
    Case { to: "French", text: "It's too big for me.", context: &["I tried on the shirt you sent me."], expect: &["grande"], why: "la chemise (fem.)" },
    Case { to: "French", text: "Did you find it?", context: &["I lost my watch at the gym."], expect: &["trouvée"], why: "la montre (fem.)" },
    Case { to: "French", text: "I'm sitting on the bank.", context: &["We went fishing this morning.", "The river is beautiful."], expect: &["rive", "berge", "bord"], why: "river bank" },
    Case { to: "Spanish", text: "They are very tired.", context: &["My two daughters just got back from camp."], expect: &["cansadas"], why: "daughters (fem. pl.)" },
    Case { to: "Spanish", text: "Is it ready yet?", context: &["I ordered a pizza an hour ago."], expect: &["lista"], why: "la pizza (fem.)" },
    Case { to: "Italian", text: "Are you ready?", context: &["Maria, the taxi is here."], expect: &["pronta"], why: "addressee is Maria" },
    Case { to: "Portuguese", text: "Are you tired?", context: &["Ana, you worked all day today."], expect: &["cansada"], why: "addressee is Ana" },
    Case { to: "German", text: "Where did you put it?", context: &["I can't find the car key."], expect: &["ihn"], why: "der Schlüssel (masc.)" },
    Case { to: "German", text: "Can you send me the report by Friday?", context: &["Dear Mrs. Weber, thank you for the meeting yesterday."], expect: &["sie ", "ihnen"], why: "formal Sie" },
    Case { to: "Russian", text: "Where did you go yesterday?", context: &["Hi Anna! I missed you at work."], expect: &["ходила", "ездила", "была"], why: "addressee is Anna (fem. past)" },
    Case { to: "French", text: "Good morning, how are you?", context: &["The meeting moved to 3pm.", "Bring the slides."], expect: &[], why: "irrelevant context: no leak" },
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: context_ab <model_dir> [--mode nf4g128] [--threads N] [--cpu]");
        std::process::exit(2);
    }
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned());
    let mode = flag("--mode").unwrap_or_else(|| "nf4g128".into());
    let threads = flag("--threads").map_or_else(kronos_2t::kernels::thread_pool::default_inference_threads, |v| v.parse().expect("--threads"));
    let mut model = ChatModel::load(&PathBuf::from(&args[1]), &mode, 2048, threads, None).expect("load model");
    if !args.iter().any(|a| a == "--cpu") {
        if let Err(e) = model.enable_metal() {
            eprintln!("metal unavailable ({e}); running on CPU");
        }
    }

    let variant: usize = flag("--prompt").map_or(0, |v| v.parse().expect("--prompt"));
    let mut tr = |text: &str, to: &str, ctx: &[String]| {
        let req = TranslateRequest { text, to, from: "English", tone: "neutral", context: ctx, temperature: 0.0 };
        if variant == 0 || ctx.is_empty() {
            return run(&mut model, &req, |_| true).map(|c| clean_translation(&c.text)).unwrap_or_else(|e| format!("ERROR {e}"));
        }
        let (_, params, prefix) = prompt(&model, &req);
        let msgs = vec![Message::new("user", &user_prompt(variant, to, ctx, text))];
        model.generate_with_tools_prefixed(&msgs, &[], prefix, &params, |_| true).map(|c| clean_translation(&c.text)).unwrap_or_else(|e| format!("ERROR {e}"))
    };

    println!("case\tto\tsource\twithout context\twith context\tcontext used\tleak-free");
    let (mut used, mut need, mut changed, mut clean) = (0, 0, 0, 0);
    for (i, c) in CASES.iter().enumerate() {
        let ctx: Vec<String> = c.context.iter().map(|s| s.to_string()).collect();
        let off = tr(c.text, c.to, &[]);
        let on = tr(c.text, c.to, &ctx);
        let low = on.to_lowercase();
        let hit = c.expect.iter().any(|e| low.contains(e));
        let off_hit = c.expect.iter().any(|e| off.to_lowercase().contains(e));
        // A leak shows up as output much longer than the context-free translation.
        let leak_free = on.chars().count() <= off.chars().count() * 2 + 10;
        if !c.expect.is_empty() {
            need += 1;
            used += hit as usize;
        }
        changed += (on != off) as usize;
        clean += leak_free as usize;
        let verdict = if c.expect.is_empty() { "n/a".to_string() } else if hit && !off_hit { "yes (context only)".into() } else if hit { "yes (both)".into() } else { "no".into() };
        println!("{}\t{}\t{}\t{}\t{}\t{} · {}\t{}", i + 1, c.to, c.text, off, on, verdict, c.why, if leak_free { "yes" } else { "NO" });
    }
    eprintln!("context used {used}/{need} · output changed {changed}/{} · leak-free {clean}/{}", CASES.len(), CASES.len());
}

/// Alternative context prompts (0 is the engine's own).
fn user_prompt(variant: usize, to: &str, ctx: &[String], text: &str) -> String {
    let lines = ctx.iter().map(|c| format!("- {c}")).collect::<Vec<_>>().join("\n");
    match variant {
        // Model card "Structured Data 2" verbatim.
        1 => format!("[Background Information]\n{lines}\n\nPlease translate the following text into {to}, taking the provided background information into consideration.\n\n[Source Text]\n{text}"),
        // Model card "Personalization".
        2 => format!("[Source Text]\n{text}\n\n[Translation Tasks]\n1. The source text is a chat message. Earlier messages in this chat:\n{lines}\n2. Use the earlier messages to choose grammatical gender, formality (tu/vous, du/Sie) and word meaning.\n3. Do not translate the earlier messages. Output only the translation.\n4. Translate the [Source Text] into {to}."),
        // Explicit hints about what to resolve.
        _ => format!("[Background Information]\nEarlier messages in this chat:\n{lines}\n\nTranslate the following text into {to}. Pronouns, gender agreement and formality must match the people and things named in the background information. Note that you should only output the translated result without any additional explanation:\n\n[Source Text]\n{text}"),
    }
}
