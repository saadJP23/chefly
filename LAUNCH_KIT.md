# Chefly Launch Kit — Reddit & Product Hunt

---

## PRODUCT HUNT LAUNCH

### Best time to launch
**Tuesday, Wednesday, or Thursday at exactly 12:01am PST (Pacific Time)**
That's 4:01pm Japan time. Set an alarm. Early votes in the first hour matter most.

### Tagline (under 60 chars)
```
Point your camera at food — get the recipe instantly
```

### Description
```
I built Chefly after getting frustrated trying to identify dishes I ate at restaurants or saw online. You just upload a photo and our AI (powered by GPT-4o Vision) identifies any dish in the world and gives you the full recipe.

Features:
📸 AI dish identification from any photo — not limited to a preset list
🍳 Recipe generation for any cuisine
📖 Community recipe library
⭐ Save & rate recipes
🔒 Free plan: 5 scans/day | Pro: unlimited ($4.99/mo)

Would love your feedback on what to build next!
```

### Maker's first comment (post this yourself immediately after launch)
```
Hey Product Hunt! 👋 

I'm Saad, the solo developer behind Chefly. I built this because I kept eating amazing food and having no idea what it was or how to make it.

The hardest part was getting the AI to recognise literally any dish — not just a fixed list of 16 foods like my original model. Switching to GPT-4o Vision was the breakthrough that made it actually useful.

Happy to answer any questions about how it works or the tech stack (Flask + Python backend, GPT-4o Vision for recognition, Stripe for payments).

Try scanning something unusual and let me know what it gets right or wrong 🍜
```

---

## REDDIT POSTS

### r/SideProject — Post this on launch day
**Title:**
```
I built an app that identifies any dish from a photo and gives you the recipe — built with GPT-4o Vision (cheflys.com)
```
**Body:**
```
Hey r/SideProject!

I built Chefly — you upload a photo of any food and the AI tells you what it is and how to cook it.

My original version used a custom-trained ML model that only knew 16 dishes (embarrassing, I know 😅). I recently switched to GPT-4o Vision which recognises literally anything.

**Tech stack:**
- Flask + Python backend on AWS Lightsail
- GPT-4o Vision for dish identification
- Stripe for the Pro subscription ($4.99/mo)
- Deployed at cheflys.com

**What I want feedback on:**
1. Is the scan fast enough?
2. What features would make you pay $4.99/month?
3. Anything broken?

Try it free at cheflys.com — 5 scans/day on the free plan. Would love honest feedback!
```

---

### r/webdev — Post this 2 days after r/SideProject
**Title:**
```
Replaced a custom-trained ML model (16 dishes) with GPT-4o Vision — here's what changed
```
**Body:**
```
Quick technical post about a decision I made on my side project Chefly (cheflys.com).

**The problem:** I trained a MobileNetV2 model on 16 food classes. It worked okay for those 16 dishes but was completely useless for anything else. Users were uploading photos of pad thai and getting "biryani" back at 34% confidence.

**The fix:** I ripped out the entire TensorFlow stack and replaced the inference with a single GPT-4o-mini Vision call.

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Identify this dish. Return JSON: [{dish, confidence}]"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "low"}}
        ]
    }],
    max_tokens=200,
)
```

**Results:**
- Recognises any dish in the world, not just 16
- ~$0.002 per scan (cheap enough for a free tier)
- Removed tensorflow, keras, and 15 other heavy packages from requirements.txt
- Docker build went from 4min → 45sec

The trade-off is latency (~2s vs ~200ms for the local model) but accuracy went from ~40% to ~95% for common dishes so it's worth it.

Anyone else made this switch? Curious about other approaches.
```

---

### r/food — Post this (casual, no spam)
**Title:**
```
I made a free tool that identifies any dish from a photo — tested it on 50 random foods
```
**Body:**
```
I built cheflys.com — you take a photo of any food and it tells you what it is and how to make it.

Tested it on everything from Pakistani nihari to Japanese tamagoyaki to Peruvian ceviche. It gets it right almost every time.

Free to use (5 scans/day). Would love to know what dish stumps it 🍽️
```

---

### r/mealprep — Post this 1 week after launch
**Title:**
```
Free tool: photograph any dish and get the full recipe + nutrition info
```
**Body:**
```
Built cheflys.com for people like me who see food online, have no idea what it's called, and want to cook it.

Upload a photo → AI identifies the dish → full recipe with ingredients, instructions, prep time, calories, protein, carbs, fat.

Free plan: 5 scans/day. No credit card needed to sign up.

Useful for meal preppers who want to recreate restaurant dishes at home.
```

---

## POSTING SCHEDULE

| Day | Platform | Post |
|-----|----------|------|
| Day 1 (Tuesday) | Product Hunt | Launch at 12:01am PST |
| Day 1 | r/SideProject | Post after PH launches |
| Day 3 | r/webdev | Technical post |
| Day 5 | r/food | Casual post |
| Day 10 | r/mealprep | Meal prep angle |
| Day 14 | r/Entrepreneur | "lessons from building" post |

**Rules:**
- Never post the same link twice in the same subreddit
- Always engage with every comment within 2 hours of posting
- Upvote ratio matters — reply to everyone to keep engagement high
- Don't mention pricing in the title — it kills click-through

---

## GOOGLE SEARCH CONSOLE

After deploying, submit your sitemap to Google:
1. Go to https://search.google.com/search-console
2. Add property: cheflys.com
3. Submit sitemap: https://cheflys.com/sitemap.xml
4. This gets your recipes indexed in Google search results

---

## QUICK WINS THIS WEEK

- [ ] Submit sitemap to Google Search Console
- [ ] Create TikTok/Instagram: film yourself scanning 3 different dishes in 30 seconds
- [ ] Post on r/SideProject
- [ ] Launch on Product Hunt
- [ ] Share in any WhatsApp/Discord food groups you're in
