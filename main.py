import telebot, pandas as pd, feedparser, requests, os, time, threading
from datetime import datetime

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
bot = telebot.TeleBot(TOKEN)

# १. FII/DII → NSDL (7:00 वाजता अपडेट)
def get_fiidii():
    try:
        url = "https://www.fpi.nsdl.co.in/web/Reports/Latest.aspx"
        df = pd.read_html(url)[0]
        row = df.iloc[0]
        return f"FII: *₹{row[3]:,.0f} Cr*\nDII: *₹{row[4]:,.0f} Cr*"
    except:
        return "FII/DII लोड होतंय... 7:30 नंतर बघ"

# २. Sensex/Nifty → Yahoo (24×7)
def get_indices():
    try:
        yf = requests.get("https://query1.finance.yahoo.com/v7/finance/quote?symbols=^BSESN,^NSEI").json()
        sx = yf['quoteResponse']['result'][0]['regularMarketPrice']
        sp = yf['quoteResponse']['result'][0]['regularMarketChangePercent']
        nf = yf['quoteResponse']['result'][1]['regularMarketPrice']
        np = yf['quoteResponse']['result'][1]['regularMarketChangePercent']
        return f"Sensex: *{sx:,.0f}* ({sp:+.2f}%)\nNifty: *{nf:,.0f}* ({np:+.2f}%)"
    except:
        return "Indices लोड होतंय..."

# ३. News → Moneycontrol (छान फॉरमॅट)
def get_news():
    feed = feedparser.parse("https://www.moneycontrol.com/news/rss")
    msg = "टॉप ३ Sensex/Nifty न्यूज\n\n"
    for e in feed.entries[:3]:
        title = e.title[:70] + "..." if len(e.title) > 70 else e.title
        msg += f"{title}\n{e.link}\n\n"
    return msg

# कमांड्स
@bot.message_handler(commands=['start'])
def start(m):
    bot.reply_to(m, "Bot LIVE!\n/fiidii → FII/DII\n/sensex → Sensex+Nifty\n/news → न्यूज")

@bot.message_handler(commands=['fiidii','sensex','news'])
def handle(m):
    cmd = m.text[1:].split()[0]
    if cmd == 'fiidii': bot.reply_to(m, get_fiidii(), parse_mode='Markdown')
    elif cmd == 'sensex': bot.reply_to(m, get_indices(), parse_mode='Markdown')
    elif cmd == 'news': bot.reply_to(m, get_news())

# रोज 7:30 PM ऑटो
def daily():
    while True:
        now = datetime.now()
        if now.hour == 19 and now.minute == 30:
            msg = f"आजचा अपडेट ({now.strftime('%d %b')})\n\n"
            msg += get_fiidii() + "\n\n"
            msg += get_indices() + "\n\n"
            msg += get_news()
            bot.send_message(CHAT_ID, msg, parse_mode='Markdown', disable_web_page_preview=True)
            print("7:30 चा मेसेज पाठवला!")
            time.sleep(70)
        time.sleep(30)

if __name__ == "__main__":
    threading.Thread(target=daily, daemon=True).start()
    print("Bot 24×7 चालू! /start कर")
    bot.infinity_polling()
