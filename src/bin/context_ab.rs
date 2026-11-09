//! Translates sentences whose correct form depends on earlier chat messages, once with that
//! context and once without, and checks the context-only forms (gender, formality, word sense).
//! Also checks context never leaks into the output.
//!
//! Usage: `context_ab <model_dir> [--mode nf4g128] [--threads N] [--cpu]`

use kronos_2t::runtime::chat::ChatModel;
use kronos_2t::runtime::translate::{clean_translation, run, TranslateRequest};
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
    // The keyboard keeps the sent translations, so real context is in the target language.
    Case { to: "French", text: "It's too big for me.", context: &["J'ai essayé la chemise que tu m'as envoyée."], expect: &["grande"], why: "FR context · la chemise" },
    Case { to: "French", text: "Did you find it?", context: &["J'ai perdu ma montre à la salle de sport."], expect: &["trouvée"], why: "FR context · la montre" },
    Case { to: "Spanish", text: "They are very tired.", context: &["Mis dos hijas acaban de volver del campamento."], expect: &["cansadas"], why: "ES context · hijas" },
    Case { to: "Italian", text: "Are you ready?", context: &["Maria, il taxi è arrivato."], expect: &["pronta"], why: "IT context · Maria" },
    Case { to: "German", text: "Can you send me the report by Friday?", context: &["Sehr geehrte Frau Weber, vielen Dank für das Treffen gestern."], expect: &["sie ", "ihnen"], why: "DE context · formal" },
    Case { to: "French", text: "Good morning, how are you?", context: &["La réunion a été déplacée à 15h.", "Apporte les diapositives."], expect: &[], why: "FR context: no echo" },
    Case { to: "French", text: "yes! see you there, can't wait", context: &["On se voit demain ? J'ai réservé pour 19h"], expect: &[], why: "FR context: past echo regression" },
    Case { to: "Spanish", text: "ok see you there", context: &["¿Vienes a la fiesta de Ana esta noche?", "Empieza a las ocho."], expect: &[], why: "ES context: no echo" },
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

    let mut tr = |text: &str, to: &str, ctx: &[String]| {
        let req = TranslateRequest { text, to, from: "English", tone: "neutral", context: ctx, temperature: 0.0 };
        run(&mut model, &req, |_| true).map(|c| clean_translation(&c.text)).unwrap_or_else(|e| format!("ERROR {e}"))
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
        // A leak shows up as a context line echoed or output much longer than without context.
        let echoed = c.context.iter().any(|l| {
            let head: String = l.to_lowercase().chars().take(14).collect();
            low.contains(head.trim())
        });
        let leak_free = !echoed && on.chars().count() <= off.chars().count() * 2 + 10;
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

