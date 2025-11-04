import telebot, pandas as pd, feedparser, requests, os, time, threading
from datetime import datetime

# तुझ्या नावानुसारच घेतो!
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# चेक करतो की टोकन आलाय का
if not TOKEN or not CHAT_ID:
    print("ERROR: TELEGRAM_BOT_TOKEN किंवा TELEGRAM_CHAT_ID सेट करा!")
    exit()
else:
    print("TOKEN & CHAT_ID मिळाले! Bot चालू...")

bot = telebot.TeleBot(TOKEN)

# NSE सेशन
s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/"})

# FII/DII
def get_fiidii():
    try:
        d = datetime.now().strftime('%d-%b-%Y').upper()
        url = f"https://www.nseindia.com/archives/equities/fiidii/{d}.csv"
        r = s.get(url, timeout=10)
        if r.status_code != 200: return "डेटा येणार नाही (7:45 नंतर बघ)"
        df = pd.read_csv(r.text.splitlines())
        f = df.iloc[0]
        return f"FII: *₹{f['FII Net (Cr)']:,.0f} Cr*\nDII: *₹{f['DII Net (Cr)']:,.0f} Cr*"
    except: return "FII/DII Error"

# Sensex + Nifty
def get_indices():
    try:
        j1 = s.get("https://www.nseindia.com/api/quote-equity?symbol=%5ENSEI").json()
        sx = j1['priceInfo']['lastPrice']
        sp = j1['priceInfo']['pChange']
        j2 = s.get("https://www.nseindia.com/api/quote-equity?symbol=%5ENIFTY%2050").json()
        nf = j2['priceInfo']['lastPrice']
        np = j2['priceInfo']['pChange']
        return f"Sensex: {sx:,.0f} ({sp:+.2f}%)\nNifty: {nf:,.0f} ({np:+.2f}%)"
    except: return "Indices बंद (मार्केट 3:30 वाजता बंद)"

# न्यूज
def get_news():
    feed = feedparser.parse("https://www.moneycontrol.com/news/rss")
    msg = "टॉप ३ न्यूज\n\n"
    for e in feed.entries[:3]:
        msg += f"• {e.title}\n{e.link}\n\n"
    return msg

# कमांड्स
@bot.message_handler(commands=['start'])
def start(m):
    bot.reply_to(m, "Bot चालू!\n/fiidii\n/sensex\n/news")

@bot.message_handler(commands=['fiidii'])
def fiidii(m): bot.reply_to(m, get_fiidii(), parse_mode='Markdown')

@bot.message_handler(commands=['sensex'])
def sensex(m): bot.reply_to(m, get_indices(), parse_mode='Markdown')

@bot.message_handler(commands=['news'])
def news(m): bot.reply_to(m, get_news())

# रोज 7:45 PM ऑटो
def daily_report():
    while True:
        now = datetime.now()
        if now.hour == 19 and now.minute == 45:
            msg = f"आजचा अपडेट ({now.strftime('%d %b')})\n\n"
            msg += get_fiidii() + "\n\n"
            msg += get_indices() + "\n\n"
            msg += get_news()
            bot.send_message(CHAT_ID, msg, parse_mode='Markdown', disable_web_page_preview=True)
            print("7:45 PM चा मेसेज पाठवला!")
            time.sleep(70)
        time.sleep(30)

# चालू कर
if __name__ == "__main__":
    threading.Thread(target=daily_report, daemon=True).start()
    print("Bot 24×7 चालू! /start करून ट्राय कर")
    bot.infinity_polling()
