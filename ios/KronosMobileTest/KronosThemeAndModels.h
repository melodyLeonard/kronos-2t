#ifndef KRONOS_THEME_AND_MODELS_H
#define KRONOS_THEME_AND_MODELS_H

#import <UIKit/UIKit.h>
#import <Vision/Vision.h>
#import <PDFKit/PDFKit.h>
#import <Photos/Photos.h>
#import <Contacts/Contacts.h>
#import <EventKit/EventKit.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#include <mach/mach_time.h>
#include "../../include/kronos_ffi.h"

static inline double kronos_monotonic_ms(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) {
        mach_timebase_info(&tb);
    }
    uint64_t ticks = mach_absolute_time();
    return ((double)ticks * (double)tb.numer / (double)tb.denom) / 1e6;
}

// Adaptive colors matching Kronos Web (light #fafaf9 / dark #0d0e11 + #131418 sidebar rail).
static inline UIColor *kronosDynamicColor(uint32_t lightHex, CGFloat lightAlpha,
                                          uint32_t darkHex, CGFloat darkAlpha) {
    return [UIColor colorWithDynamicProvider:^UIColor *(UITraitCollection *tc) {
        BOOL isDark = (tc.userInterfaceStyle == UIUserInterfaceStyleDark);
        uint32_t hex = isDark ? darkHex : lightHex;
        CGFloat a = isDark ? darkAlpha : lightAlpha;
        CGFloat r = ((hex >> 16) & 0xFF) / 255.0;
        CGFloat g = ((hex >> 8) & 0xFF) / 255.0;
        CGFloat b = (hex & 0xFF) / 255.0;
        return [UIColor colorWithRed:r green:g blue:b alpha:a];
    }];
}

static inline UIColor *kronosBgColor(void)       { return kronosDynamicColor(0xFAFAF9, 1.0, 0x0D0E11, 1.0); }
static inline UIColor *kronosPanelColor(void)    { return kronosDynamicColor(0xFFFFFF, 1.0, 0x15171C, 1.0); }
static inline UIColor *kronosSubtleColor(void)   { return kronosDynamicColor(0xF3F3F1, 1.0, 0x1D2026, 1.0); }
static inline UIColor *kronosTextColor(void)     { return kronosDynamicColor(0x18181B, 1.0, 0xF4F4F5, 1.0); }
static inline UIColor *kronosMutedColor(void)    { return kronosDynamicColor(0x71717A, 1.0, 0xA1A1AA, 1.0); }
static inline UIColor *kronosLineColor(void)     { return kronosDynamicColor(0xE4E4E7, 1.0, 0x272A32, 1.0); }
static inline UIColor *kronosAccentColor(void)   { return kronosDynamicColor(0x0D9488, 1.0, 0x14B8A6, 1.0); }
static inline UIColor *kronosUserBubbleBg(void)  { return kronosDynamicColor(0x0D9488, 1.0, 0x0F766E, 1.0); }

static inline UIColor *kronosSidebarBg(void)     { return [UIColor colorWithRed:0.075 green:0.078 blue:0.094 alpha:1.0]; }
static inline UIColor *kronosSidebarPanel(void)  { return [UIColor colorWithRed:0.110 green:0.118 blue:0.141 alpha:1.0]; }
static inline UIColor *kronosSidebarActive(void) { return [UIColor colorWithRed:0.165 green:0.176 blue:0.216 alpha:1.0]; }
static inline UIColor *kronosSidebarText(void)   { return [UIColor colorWithRed:0.925 green:0.925 blue:0.937 alpha:1.0]; }
static inline UIColor *kronosSidebarMuted(void)  { return [UIColor colorWithRed:0.557 green:0.573 blue:0.624 alpha:1.0]; }

#pragma mark - Tactile controls

/// A button that answers every touch: it shrinks and dims while pressed, springs back on release,
/// taps the haptic engine, can flash a confirmation symbol, and can pulse while an action runs.
@interface KronosTapButton : UIButton
@property (nonatomic, assign) BOOL pulsing;
@property (nonatomic, strong) UIImpactFeedbackGenerator *haptic;
+ (instancetype)tapButton;
- (void)flashSymbol:(NSString *)symbolName tint:(UIColor *)tint;
- (void)bounce;
@end

@implementation KronosTapButton
+ (instancetype)tapButton {
    KronosTapButton *b = [[KronosTapButton alloc] initWithFrame:CGRectZero];
    return b;
}
- (instancetype)initWithFrame:(CGRect)frame {
    self = [super initWithFrame:frame];
    if (self) {
        self.haptic = [[UIImpactFeedbackGenerator alloc] initWithStyle:UIImpactFeedbackStyleLight];
        [self addTarget:self action:@selector(kronosPressDown) forControlEvents:UIControlEventTouchDown | UIControlEventTouchDragEnter];
        [self addTarget:self action:@selector(kronosPressUp) forControlEvents:UIControlEventTouchUpInside | UIControlEventTouchUpOutside | UIControlEventTouchCancel | UIControlEventTouchDragExit];
        [self addTarget:self action:@selector(kronosTapped) forControlEvents:UIControlEventTouchUpInside];
    }
    return self;
}
- (void)kronosPressDown {
    [self.haptic prepare];
    [UIView animateWithDuration:0.09 delay:0 options:UIViewAnimationOptionCurveEaseOut | UIViewAnimationOptionAllowUserInteraction animations:^{
        self.transform = CGAffineTransformMakeScale(0.88, 0.88);
        self.alpha = 0.72;
    } completion:nil];
}
- (void)kronosPressUp {
    [UIView animateWithDuration:0.32 delay:0 usingSpringWithDamping:0.55 initialSpringVelocity:4.0
                        options:UIViewAnimationOptionAllowUserInteraction animations:^{
        self.transform = CGAffineTransformIdentity;
        self.alpha = 1.0;
    } completion:nil];
}
- (void)kronosTapped {
    [self.haptic impactOccurred];
}
- (void)bounce {
    [UIView animateWithDuration:0.1 animations:^{ self.transform = CGAffineTransformMakeScale(1.18, 1.18); } completion:^(BOOL f) {
        [UIView animateWithDuration:0.3 delay:0 usingSpringWithDamping:0.5 initialSpringVelocity:3.0 options:0 animations:^{
            self.transform = CGAffineTransformIdentity;
        } completion:nil];
    }];
}
/// Shows `symbolName` in `tint` for a moment (a checkmark after copying), then restores the icon.
- (void)flashSymbol:(NSString *)symbolName tint:(UIColor *)tint {
    UIImage *restore = [self imageForState:UIControlStateNormal];
    UIColor *restoreTint = self.tintColor;
    UIImageSymbolConfiguration *cfg = restore.symbolConfiguration ?: [UIImageSymbolConfiguration configurationWithPointSize:15.0 weight:UIImageSymbolWeightMedium];
    [self setImage:[UIImage systemImageNamed:symbolName withConfiguration:cfg] forState:UIControlStateNormal];
    self.tintColor = tint;
    [self bounce];
    __weak typeof(self) weakSelf = self;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(1.8 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
        __strong typeof(weakSelf) me = weakSelf;
        if (me == nil) return;
        [UIView transitionWithView:me duration:0.2 options:UIViewAnimationOptionTransitionCrossDissolve animations:^{
            [me setImage:restore forState:UIControlStateNormal];
            me.tintColor = restoreTint;
        } completion:nil];
    });
}
- (void)setPulsing:(BOOL)pulsing {
    if (_pulsing == pulsing) return;
    _pulsing = pulsing;
    [self.layer removeAnimationForKey:@"kronosPulse"];
    if (pulsing) {
        CABasicAnimation *a = [CABasicAnimation animationWithKeyPath:@"transform.scale"];
        a.fromValue = @1.0; a.toValue = @1.16; a.duration = 0.55; a.autoreverses = YES; a.repeatCount = HUGE_VALF;
        a.timingFunction = [CAMediaTimingFunction functionWithName:kCAMediaTimingFunctionEaseInEaseOut];
        [self.layer addAnimation:a forKey:@"kronosPulse"];
    }
}
@end

/// A tappable row (sidebar footer, cards) with the same press feedback as `KronosTapButton`.
@interface KronosPressControl : UIControl
@end
@implementation KronosPressControl
- (instancetype)initWithFrame:(CGRect)frame {
    self = [super initWithFrame:frame];
    if (self) {
        [self addTarget:self action:@selector(kronosDown) forControlEvents:UIControlEventTouchDown | UIControlEventTouchDragEnter];
        [self addTarget:self action:@selector(kronosUp) forControlEvents:UIControlEventTouchUpInside | UIControlEventTouchUpOutside | UIControlEventTouchCancel | UIControlEventTouchDragExit];
        [self addTarget:self action:@selector(kronosHaptic) forControlEvents:UIControlEventTouchUpInside];
    }
    return self;
}
- (void)kronosDown {
    [UIView animateWithDuration:0.09 delay:0 options:UIViewAnimationOptionAllowUserInteraction animations:^{
        self.transform = CGAffineTransformMakeScale(0.97, 0.97); self.alpha = 0.8;
    } completion:nil];
}
- (void)kronosUp {
    [UIView animateWithDuration:0.28 delay:0 usingSpringWithDamping:0.6 initialSpringVelocity:3.0 options:UIViewAnimationOptionAllowUserInteraction animations:^{
        self.transform = CGAffineTransformIdentity; self.alpha = 1.0;
    } completion:nil];
}
- (void)kronosHaptic {
    [[[UIImpactFeedbackGenerator alloc] initWithStyle:UIImpactFeedbackStyleLight] impactOccurred];
}
@end

/// Scrolling bar waveform of recent microphone levels (0..1), newest on the right.
@interface KronosLevelWaveView : UIView
@property (nonatomic, strong) NSMutableArray<NSNumber *> *levels;
@property (nonatomic, assign) NSUInteger capacity;
- (void)pushLevel:(float)level;
- (void)reset;
@end
@implementation KronosLevelWaveView
- (instancetype)initWithFrame:(CGRect)frame {
    self = [super initWithFrame:frame];
    if (self) {
        _levels = [NSMutableArray array];
        _capacity = 44;
        self.backgroundColor = [UIColor clearColor];
        self.contentMode = UIViewContentModeRedraw;
    }
    return self;
}
- (void)pushLevel:(float)level {
    [self.levels addObject:@(MAX(0.0f, MIN(1.0f, level)))];
    while (self.levels.count > self.capacity) [self.levels removeObjectAtIndex:0];
    [self setNeedsDisplay];
}
- (void)reset {
    [self.levels removeAllObjects];
    [self setNeedsDisplay];
}
- (void)drawRect:(CGRect)rect {
    CGContextRef ctx = UIGraphicsGetCurrentContext();
    CGFloat w = self.bounds.size.width, h = self.bounds.size.height;
    CGFloat step = w / (CGFloat)self.capacity;
    CGFloat barW = MAX(2.0, step * 0.55);
    NSUInteger n = self.levels.count;
    [self.tintColor setFill];
    for (NSUInteger i = 0; i < self.capacity; i++) {
        float lv = 0.0f;
        NSInteger idx = (NSInteger)i - (NSInteger)(self.capacity - n);
        if (idx >= 0) lv = self.levels[(NSUInteger)idx].floatValue;
        CGFloat bh = MAX(3.0, lv * h);
        CGRect r = CGRectMake(i * step + (step - barW) / 2.0, (h - bh) / 2.0, barW, bh);
        [[UIBezierPath bezierPathWithRoundedRect:r cornerRadius:barW / 2.0] fill];
    }
    (void)ctx;
}
@end

/// A short pill at the bottom of `host` ("Copied"), fading after 1.4 s.
static inline void kronosShowToast(UIView *host, NSString *text) {
    if (host == nil) return;
    UILabel *pill = [[UILabel alloc] init];
    pill.translatesAutoresizingMaskIntoConstraints = NO;
    pill.text = text;
    pill.font = [UIFont systemFontOfSize:13.0 weight:UIFontWeightMedium];
    pill.textColor = [UIColor whiteColor];
    pill.textAlignment = NSTextAlignmentCenter;
    pill.backgroundColor = [UIColor colorWithWhite:0.1 alpha:0.92];
    pill.layer.cornerRadius = 15.0;
    pill.clipsToBounds = YES;
    pill.alpha = 0.0;
    [host addSubview:pill];
    [NSLayoutConstraint activateConstraints:@[
        [pill.centerXAnchor constraintEqualToAnchor:host.centerXAnchor],
        [pill.bottomAnchor constraintEqualToAnchor:host.safeAreaLayoutGuide.bottomAnchor constant:-84.0],
        [pill.heightAnchor constraintEqualToConstant:30.0],
        [pill.widthAnchor constraintGreaterThanOrEqualToConstant:84.0],
    ]];
    pill.transform = CGAffineTransformMakeTranslation(0, 10);
    [UIView animateWithDuration:0.18 animations:^{ pill.alpha = 1.0; pill.transform = CGAffineTransformIdentity; }];
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(2.0 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
        [UIView animateWithDuration:0.22 animations:^{ pill.alpha = 0.0; pill.transform = CGAffineTransformMakeTranslation(0, 8); }
                         completion:^(BOOL f) { [pill removeFromSuperview]; }];
    });
}

static NSString *const kKronosCreatorProfileFacts =
    @"[Verified Public Profile — Use ONLY these exact facts. Pronouns for Melody Leonard are strictly he/him/his. He is a Software Engineer, NEVER a musician, songwriter, or actor.]\n"
     "- Full name: Melody Leonard (Melody Leonard Oparaugo), pronouns: he / him / his\n"
     "- Profession & experience: Senior Software Engineer and Systems Architect with over 10 years of software engineering experience (@IPMScoutek). He created and built Kronos.\n"
     "- His GitHub: https://github.com/melodyLeonard (52+ public repositories including kronos-2t, tshare, t-share, cross-native, gpt_2_tokenizer, tomatoe, codejam-admin-platform, helloworld-grpc, rust-tcp, identicon, medium_graphql_api, token-swap)\n"
     "- His LinkedIn: https://www.linkedin.com/in/melody-leonard-a362b6192\n"
     "- His Email: melodyleonard7@gmail.com\n"
     "- His X / Twitter: https://x.com/Melody_Leonard7 (@Melody_Leonard7)\n"
     "- His engineering specialties: distributed systems, event-driven microservices (Apache Kafka, CQRS, GraphQL, gRPC, REST), Rust systems and local Apple Silicon inference engines (Kronos / kronos-2t), database storage internals (PostgreSQL, MySQL, MongoDB, Neo4j, Redis), offline-first mobile and React Native apps, and zero-touch CI/CD and App Store / TestFlight pipelines (AWS, Docker, Kubernetes).\n"
     "- His languages & frameworks: Rust, TypeScript, JavaScript, Elixir, Solidity, C, Node.js, NestJS, Express, React.js, Next.js, React Native, Tailwind.";

static NSString *const kKronosMobileSystemPrompt =
    @"You are Kronos, a local personal software system running directly on this iPhone. "
     "Your creator and builder is Melody Leonard (full name: Melody Leonard Oparaugo, pronouns: he/him/his). "
     "Melody Leonard (he/him/his) is a male Senior Software Engineer and Systems Architect with over 10 years of software engineering experience (@IPMScoutek). "
     "Keep role boundaries clear: you are Kronos, and Melody Leonard is the human software engineer who created and built you. Never say 'I am Melody Leonard', 'I created Kronos', or 'I made Kronos', and never repeat the user's 'my' as 'My ...'. "
     "Melody Leonard is a software engineer, never a musician, songwriter, or actor, and never refer to him with she/her pronouns. "
     "His public links and contact: GitHub https://github.com/melodyLeonard, LinkedIn https://www.linkedin.com/in/melody-leonard-a362b6192, Email melodyleonard7@gmail.com, X @Melody_Leonard7. "
     "His stack: Rust, TypeScript, JavaScript, Elixir, Solidity, C, Node.js, NestJS, Apache Kafka, GraphQL, gRPC, PostgreSQL, MongoDB, Neo4j, Redis, React.js, Next.js, React Native, AWS, Docker, Kubernetes. "
     "When describing an attached photograph of a physical object or scene, identify the physical objects and setting naturally and concisely without calling it a text detection or OCR task. "
     "Answer strictly from verified context and never fabricate placeholder numbers or messages. "
     "When an iPhone Device Context block is present, trust its access lines and never claim to lack access it lists. "
     "You cannot see which apps are installed: never state that an app is or is not installed.";

/// Bearer token for the Mac bridge, compiled in from ~/.kronos/device_token.
static inline void kronosApplyMacAuth(NSMutableURLRequest *req) {
#ifdef KRONOS_MAC_TOKEN
    [req setValue:@"Bearer " KRONOS_MAC_TOKEN forHTTPHeaderField:@"Authorization"];
#else
    (void)req;
#endif
}

static inline NSString *kronosDocumentsDirectory(void) {
    NSArray<NSString *> *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    return paths.firstObject ?: NSTemporaryDirectory();
}

static inline NSString *kronosWorkspaceDirectory(void) {
    NSString *dir = [kronosDocumentsDirectory() stringByAppendingPathComponent:@"KronosWorkspace"];
    [[NSFileManager defaultManager] createDirectoryAtPath:dir
                              withIntermediateDirectories:YES
                                               attributes:nil
                                                    error:nil];
    return dir;
}

static inline NSString *kronosFormatBytes(NSUInteger bytes) {
    if (bytes < 1024) return [NSString stringWithFormat:@"%lu B", (unsigned long)bytes];
    if (bytes < 1024 * 1024) return [NSString stringWithFormat:@"%.1f KB", (double)bytes / 1024.0];
    return [NSString stringWithFormat:@"%.1f MB", (double)bytes / (1024.0 * 1024.0)];
}

static inline NSString *kronosRelativeDateString(NSTimeInterval epochSecs) {
    if (epochSecs <= 0) return @"Not indexed yet";
    NSTimeInterval diff = [[NSDate date] timeIntervalSince1970] - epochSecs;
    if (diff < 60) return @"Just now";
    if (diff < 3600) return [NSString stringWithFormat:@"%dm ago", (int)(diff / 60)];
    if (diff < 86400) return [NSString stringWithFormat:@"%dh ago", (int)(diff / 3600)];
    if (diff < 172800) return @"Yesterday";
    NSDateFormatter *df = [[NSDateFormatter alloc] init];
    df.dateFormat = @"MMM d";
    return [df stringFromDate:[NSDate dateWithTimeIntervalSince1970:epochSecs]];
}

// Persistent Mobile Settings (Develop Device Context, Allow Phone Use, Local Mac Sync, Voice).
@interface KronosMobileSettings : NSObject
@property (nonatomic, assign) BOOL developDeviceContext;
@property (nonatomic, assign) BOOL allowPhoneUse;
@property (nonatomic, assign) BOOL macSyncEnabled;
@property (nonatomic, assign) BOOL preferMacEngineOnLan;
@property (nonatomic, copy) NSString *macServerURL;
@property (nonatomic, assign) BOOL spokenRepliesEnabled;
@property (nonatomic, assign) BOOL autoSendVoiceAfterPause;
+ (instancetype)sharedSettings;
- (void)save;
@end

@implementation KronosMobileSettings
+ (instancetype)sharedSettings {
    static KronosMobileSettings *inst = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        inst = [[KronosMobileSettings alloc] init];
    });
    return inst;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        NSUserDefaults *ud = [NSUserDefaults standardUserDefaults];
        _developDeviceContext = ([ud objectForKey:@"kronos_develop_device_ctx"] != nil)
            ? [ud boolForKey:@"kronos_develop_device_ctx"] : YES;
        _allowPhoneUse = ([ud objectForKey:@"kronos_allow_phone_use"] != nil)
            ? [ud boolForKey:@"kronos_allow_phone_use"] : YES;
        _macSyncEnabled = ([ud objectForKey:@"kronos_mac_sync_enabled"] != nil)
            ? [ud boolForKey:@"kronos_mac_sync_enabled"] : YES;
        _preferMacEngineOnLan = ([ud objectForKey:@"kronos_prefer_mac_14b"] != nil)
            ? [ud boolForKey:@"kronos_prefer_mac_14b"] : YES;
        NSString *savedURL = [ud stringForKey:@"kronos_mac_server_url"];
        _macServerURL = (savedURL.length > 0) ? savedURL : @"http://192.168.1.70:18080";
        _spokenRepliesEnabled = [ud boolForKey:@"kronos_spoken_replies"];
        _autoSendVoiceAfterPause = ([ud objectForKey:@"kronos_autosend_voice"] != nil)
            ? [ud boolForKey:@"kronos_autosend_voice"] : YES;
    }
    return self;
}

- (void)save {
    NSUserDefaults *ud = [NSUserDefaults standardUserDefaults];
    [ud setBool:self.developDeviceContext forKey:@"kronos_develop_device_ctx"];
    [ud setBool:self.allowPhoneUse forKey:@"kronos_allow_phone_use"];
    [ud setBool:self.macSyncEnabled forKey:@"kronos_mac_sync_enabled"];
    [ud setBool:self.preferMacEngineOnLan forKey:@"kronos_prefer_mac_14b"];
    [ud setObject:(self.macServerURL ?: @"http://192.168.1.70:18080") forKey:@"kronos_mac_server_url"];
    [ud setBool:self.spokenRepliesEnabled forKey:@"kronos_spoken_replies"];
    [ud setBool:self.autoSendVoiceAfterPause forKey:@"kronos_autosend_voice"];
    [ud synchronize];
}
@end

static inline NSString *kronosNormalizeApostrophes(NSString *s) {
    if (s.length == 0) return @"";
    NSString *out = [s stringByReplacingOccurrencesOfString:@"\u2019" withString:@"'"];
    out = [out stringByReplacingOccurrencesOfString:@"\u2018" withString:@"'"];
    return out;
}

static inline NSArray<NSString *> *kronosSplitAlphaWords(NSString *s) {
    NSArray<NSString *> *raw = [kronosNormalizeApostrophes(s) componentsSeparatedByCharactersInSet:[[NSCharacterSet alphanumericCharacterSet] invertedSet]];
    NSMutableArray<NSString *> *out = [NSMutableArray array];
    for (NSString *w in raw) {
        if (w.length > 0) [out addObject:[w lowercaseString]];
    }
    return out;
}

static inline NSSet<NSString *> *kronosRelationshipAliasesForTarget(NSString *lowTarget, NSString **outRelationRole) {
    if ([@[@"dad", @"dads", @"daddy", @"father", @"papa", @"pops", @"pop"] containsObject:lowTarget]) {
        if (outRelationRole) *outRelationRole = @"father";
        return [NSSet setWithArray:@[@"dad", @"my dad", @"daddy", @"father", @"papa", @"pops", @"pop"]];
    }
    if ([@[@"mom", @"moms", @"mum", @"mums", @"mommy", @"mummy", @"mother", @"mama"] containsObject:lowTarget]) {
        if (outRelationRole) *outRelationRole = @"mother";
        return [NSSet setWithArray:@[@"mom", @"my mom", @"mum", @"my mum", @"mommy", @"mummy", @"mother", @"mama"]];
    }
    if ([@[@"wife", @"spouse", @"partner", @"husband"] containsObject:lowTarget]) {
        if (outRelationRole) *outRelationRole = lowTarget;
        return [NSSet setWithArray:@[lowTarget, @"spouse", @"partner"]];
    }
    if ([@[@"brother", @"bro"] containsObject:lowTarget]) {
        if (outRelationRole) *outRelationRole = @"brother";
        return [NSSet setWithArray:@[@"brother", @"bro"]];
    }
    if ([@[@"sister", @"sis"] containsObject:lowTarget]) {
        if (outRelationRole) *outRelationRole = @"sister";
        return [NSSet setWithArray:@[@"sister", @"sis"]];
    }
    if (outRelationRole) *outRelationRole = nil;
    return nil;
}

// On-Device iOS Context Indexer (Contacts, Photos, Calendar, Workspace Files, and Local Mac Companion Sync).
@interface KronosIosDeviceContextManager : NSObject
@property (nonatomic, assign) NSUInteger contactsCount;
@property (nonatomic, assign) NSUInteger photosCount;
@property (nonatomic, assign) NSUInteger todayPhotosCount;
@property (nonatomic, assign) NSUInteger calendarEventsCount;
@property (nonatomic, assign) NSUInteger workspaceFilesCount;
@property (nonatomic, assign) NSTimeInterval lastIndexedAt;
@property (nonatomic, assign) BOOL macReachable;
@property (nonatomic, assign) NSUInteger macIndexedEntries;
@property (nonatomic, assign) BOOL isRefreshing;
+ (instancetype)sharedManager;
- (void)refreshDeviceContextInteractive:(BOOL)promptPermissions completion:(void (^)(void))completion;
- (NSString *)indexSummaryText;
/// Documents found under the folders the user granted (iCloud Drive, On My iPhone), indexed by name.
@property (nonatomic, strong) NSArray<NSDictionary *> *documentIndex;
@property (nonatomic, assign) NSUInteger documentsCount;
- (NSUInteger)indexedFolderCount;
- (NSArray<NSString *> *)indexedFolderNames;
- (BOOL)addIndexedFolderURL:(NSURL *)url;
- (void)reindexDocuments;
- (NSArray<NSDictionary *> *)searchDocuments:(NSString *)term limit:(NSUInteger)limit;
/// Copies the document into `destDir` (downloading an iCloud placeholder first) and returns the local path.
- (NSString *)materializeDocument:(NSDictionary *)doc intoDirectory:(NSString *)destDir;
- (NSString *)macSyncStatusText;
- (NSString *)buildDeviceContextPromptBlockIfRelevant:(NSString *)query;
- (NSString *)resolveContactPhoneDigitsForName:(NSString *)targetName outMatchedName:(NSString **)outName;
- (void)lookupContactsForTarget:(NSString *)rawTarget
                     completion:(void (^)(BOOL found, NSString *directReply, NSString *primaryCallDigits))completion;
- (void)fetchLatestPhotoForPostWithCompletion:(void (^)(UIImage *image, NSDate *takenDate, BOOL takenToday, NSString *errorNote))completion;
/// kind: 0 photo (no screenshots), 1 screenshot, 2 video. Runs off the main queue; completes on main.
- (void)fetchLatestMediaOfKind:(NSInteger)kind completion:(void (^)(UIImage *image, NSDate *date, NSUInteger total, BOOL limited, NSString *errorNote))completion;
@end

@implementation KronosIosDeviceContextManager
+ (instancetype)sharedManager {
    static KronosIosDeviceContextManager *mgr = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        mgr = [[KronosIosDeviceContextManager alloc] init];
    });
    return mgr;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        NSUserDefaults *ud = [NSUserDefaults standardUserDefaults];
        _contactsCount = (NSUInteger)[ud integerForKey:@"kronos_idx_contacts"];
        _photosCount = (NSUInteger)[ud integerForKey:@"kronos_idx_photos"];
        _todayPhotosCount = (NSUInteger)[ud integerForKey:@"kronos_idx_today_photos"];
        _calendarEventsCount = (NSUInteger)[ud integerForKey:@"kronos_idx_cal"];
        _workspaceFilesCount = (NSUInteger)[ud integerForKey:@"kronos_idx_ws"];
        _lastIndexedAt = [ud doubleForKey:@"kronos_idx_ts"];
        _documentIndex = [ud arrayForKey:@"kronos_idx_docs"] ?: @[];
        _documentsCount = _documentIndex.count;
    }
    return self;
}

- (NSArray<NSData *> *)folderBookmarks {
    return [[NSUserDefaults standardUserDefaults] arrayForKey:@"kronos_idx_folders"] ?: @[];
}

- (NSUInteger)indexedFolderCount {
    return [self folderBookmarks].count;
}

- (NSArray<NSString *> *)indexedFolderNames {
    NSMutableArray<NSString *> *names = [NSMutableArray array];
    for (NSData *bm in [self folderBookmarks]) {
        BOOL stale = NO;
        NSURL *u = [NSURL URLByResolvingBookmarkData:bm options:NSURLBookmarkResolutionWithoutUI relativeToURL:nil bookmarkDataIsStale:&stale error:nil];
        if (u != nil) [names addObject:u.lastPathComponent];
    }
    return names;
}

- (BOOL)addIndexedFolderURL:(NSURL *)url {
    // The picker opens in the app's own folder; granting that finds nothing.
    if ([url.path containsString:NSHomeDirectory().lastPathComponent]) return NO;
    BOOL sec = [url startAccessingSecurityScopedResource];
    NSError *err = nil;
    NSData *bm = [url bookmarkDataWithOptions:0 includingResourceValuesForKeys:nil relativeToURL:nil error:&err];
    if (sec) [url stopAccessingSecurityScopedResource];
    if (bm == nil) return NO;
    NSMutableArray<NSData *> *all = [[self folderBookmarks] mutableCopy];
    for (NSData *existing in all) {
        BOOL stale = NO;
        NSURL *u = [NSURL URLByResolvingBookmarkData:existing options:NSURLBookmarkResolutionWithoutUI relativeToURL:nil bookmarkDataIsStale:&stale error:nil];
        if (u != nil && [u.path isEqualToString:url.path]) return YES;
    }
    [all addObject:bm];
    [[NSUserDefaults standardUserDefaults] setObject:all forKey:@"kronos_idx_folders"];
    return YES;
}

/// Walks every granted folder (6 levels, 20k entries) and records files, including iCloud
/// placeholders (`.name.ext.icloud`) that are not downloaded yet.
- (void)reindexDocuments {
    NSMutableArray<NSDictionary *> *docs = [NSMutableArray array];
    NSArray<NSData *> *bms = [self folderBookmarks];
    for (NSUInteger fi = 0; fi < bms.count; fi++) {
        BOOL stale = NO;
        NSURL *root = [NSURL URLByResolvingBookmarkData:bms[fi] options:NSURLBookmarkResolutionWithoutUI relativeToURL:nil bookmarkDataIsStale:&stale error:nil];
        if (root == nil) continue;
        BOOL sec = [root startAccessingSecurityScopedResource];
        NSDirectoryEnumerator<NSURL *> *en = [[NSFileManager defaultManager]
            enumeratorAtURL:root
            includingPropertiesForKeys:@[NSURLIsDirectoryKey, NSURLIsPackageKey, NSURLContentModificationDateKey, NSURLFileSizeKey]
            options:NSDirectoryEnumerationSkipsPackageDescendants
            errorHandler:^BOOL(NSURL *u, NSError *e) { (void)u; (void)e; return YES; }];
        NSUInteger rootDepth = root.pathComponents.count;
        for (NSURL *u in en) {
            if (docs.count >= 20000) break;
            if (u.pathComponents.count - rootDepth > 6) { [en skipDescendants]; continue; }
            NSString *name = u.lastPathComponent;
            if ([name isEqualToString:@".Trash"] || [name hasPrefix:@".Trash"]) { [en skipDescendants]; continue; }
            NSNumber *isDir = nil;
            [u getResourceValue:&isDir forKey:NSURLIsDirectoryKey error:nil];
            if (isDir.boolValue) continue;
            BOOL placeholder = [name hasPrefix:@"."] && [name hasSuffix:@".icloud"];
            NSString *shown = name;
            if (placeholder) {
                shown = [[name substringFromIndex:1] stringByDeletingPathExtension];
            } else if ([name hasPrefix:@"."]) {
                continue;
            }
            NSDate *mod = nil;
            [u getResourceValue:&mod forKey:NSURLContentModificationDateKey error:nil];
            NSNumber *size = nil;
            [u getResourceValue:&size forKey:NSURLFileSizeKey error:nil];
            NSString *rel = [[u.pathComponents subarrayWithRange:NSMakeRange(rootDepth, u.pathComponents.count - rootDepth)] componentsJoinedByString:@"/"];
            [docs addObject:@{
                @"name": shown,
                @"rel": rel,
                @"folder": @(fi),
                @"root": root.lastPathComponent ?: @"",
                @"placeholder": @(placeholder),
                @"size": size ?: @0,
                @"modified": @(mod ? mod.timeIntervalSince1970 : 0),
            }];
        }
        if (sec) [root stopAccessingSecurityScopedResource];
    }
    self.documentIndex = docs;
    self.documentsCount = docs.count;
    [[NSUserDefaults standardUserDefaults] setObject:docs forKey:@"kronos_idx_docs"];
}

/// Name search: every query word of 3+ letters as a substring, shorter words whole; ranked by
/// how much of the name the query covers, newest first on ties.
- (NSArray<NSDictionary *> *)searchDocuments:(NSString *)term limit:(NSUInteger)limit {
    NSMutableArray<NSDictionary *> *hits = [NSMutableArray array];
    NSCharacterSet *sep = [[NSCharacterSet alphanumericCharacterSet] invertedSet];
    NSMutableArray<NSString *> *words = [NSMutableArray array];
    for (NSString *w in [[term lowercaseString] componentsSeparatedByCharactersInSet:sep]) {
        if (w.length > 0) [words addObject:w];
    }
    if (words.count == 0) return @[];
    NSSet<NSString *> *weak = [NSSet setWithArray:@[@"result", @"results", @"file", @"files", @"document", @"documents", @"doc", @"copy", @"pdf", @"of", @"for", @"and", @"my", @"the"]];
    for (NSDictionary *d in self.documentIndex) {
        NSString *name = [d[@"name"] lowercaseString];
        NSSet<NSString *> *nameWords = [NSSet setWithArray:[name componentsSeparatedByCharactersInSet:sep]];
        NSUInteger strong = 0, matched = 0, required = 0;
        for (NSString *w in words) {
            BOOL isWeak = [weak containsObject:w];
            BOOL ok = (w.length >= 3) ? ([name rangeOfString:w].location != NSNotFound) : [nameWords containsObject:w];
            if (!isWeak) { required++; if (ok) strong++; }
            if (ok) matched++;
        }
        BOOL pass = required > 0 ? (strong == required) : (matched == words.count);
        if (!pass) continue;
        NSMutableDictionary *h = [d mutableCopy];
        h[@"score"] = @((double)matched / (double)MAX(1u, (unsigned)nameWords.count) + (double)matched);
        [hits addObject:h];
    }
    [hits sortUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
        NSComparisonResult r = [b[@"score"] compare:a[@"score"]];
        if (r != NSOrderedSame) return r;
        return [b[@"modified"] compare:a[@"modified"]];
    }];
    if (hits.count > limit) return [hits subarrayWithRange:NSMakeRange(0, limit)];
    return hits;
}

- (NSString *)materializeDocument:(NSDictionary *)doc intoDirectory:(NSString *)destDir {
    NSArray<NSData *> *bms = [self folderBookmarks];
    NSUInteger fi = [doc[@"folder"] unsignedIntegerValue];
    if (fi >= bms.count) return nil;
    BOOL stale = NO;
    NSURL *root = [NSURL URLByResolvingBookmarkData:bms[fi] options:NSURLBookmarkResolutionWithoutUI relativeToURL:nil bookmarkDataIsStale:&stale error:nil];
    if (root == nil) return nil;
    BOOL sec = [root startAccessingSecurityScopedResource];
    NSString *rel = doc[@"rel"];
    NSURL *src = [root URLByAppendingPathComponent:rel];
    NSFileManager *fm = [NSFileManager defaultManager];
    if ([doc[@"placeholder"] boolValue]) {
        // `.name.ext.icloud` -> `name.ext`; ask iCloud for the bytes and wait up to 25 s.
        NSURL *real = [[src URLByDeletingLastPathComponent] URLByAppendingPathComponent:doc[@"name"]];
        [fm startDownloadingUbiquitousItemAtURL:real error:nil];
        for (int i = 0; i < 50 && ![fm fileExistsAtPath:real.path]; i++) {
            [NSThread sleepForTimeInterval:0.5];
        }
        src = real;
    }
    NSString *dest = nil;
    if ([fm fileExistsAtPath:src.path]) {
        [fm createDirectoryAtPath:destDir withIntermediateDirectories:YES attributes:nil error:nil];
        dest = [destDir stringByAppendingPathComponent:doc[@"name"]];
        [fm removeItemAtPath:dest error:nil];
        NSError *err = nil;
        if (![fm copyItemAtURL:src toURL:[NSURL fileURLWithPath:dest] error:&err]) dest = nil;
    }
    if (sec) [root stopAccessingSecurityScopedResource];
    return dest;
}

- (void)persistCounts {
    NSUserDefaults *ud = [NSUserDefaults standardUserDefaults];
    [ud setInteger:(NSInteger)self.contactsCount forKey:@"kronos_idx_contacts"];
    [ud setInteger:(NSInteger)self.photosCount forKey:@"kronos_idx_photos"];
    [ud setInteger:(NSInteger)self.todayPhotosCount forKey:@"kronos_idx_today_photos"];
    [ud setInteger:(NSInteger)self.calendarEventsCount forKey:@"kronos_idx_cal"];
    [ud setInteger:(NSInteger)self.workspaceFilesCount forKey:@"kronos_idx_ws"];
    [ud setDouble:self.lastIndexedAt forKey:@"kronos_idx_ts"];
}

- (NSString *)indexSummaryText {
    if (![KronosMobileSettings sharedSettings].developDeviceContext) {
        return @"Paused · Enable Develop Device Context above";
    }
    if (self.isRefreshing) {
        return @"Indexing on-device context...";
    }
    return [NSString stringWithFormat:@"%lu contacts · %lu photos (%lu today) · %lu documents",
            (unsigned long)self.contactsCount,
            (unsigned long)self.photosCount,
            (unsigned long)self.todayPhotosCount,
            (unsigned long)(self.workspaceFilesCount + self.documentsCount)];
}

- (NSString *)macSyncStatusText {
    KronosMobileSettings *s = [KronosMobileSettings sharedSettings];
    if (!s.macSyncEnabled) return @"Disabled";
    if (self.macReachable) {
        return [NSString stringWithFormat:@"Connected · %lu Mac items indexed", (unsigned long)self.macIndexedEntries];
    }
    return [NSString stringWithFormat:@"Standby (%@)", s.macServerURL];
}

- (void)refreshDeviceContextInteractive:(BOOL)promptPermissions completion:(void (^)(void))completion {
    if (self.isRefreshing) {
        if (completion) completion();
        return;
    }
    self.isRefreshing = YES;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        NSArray<NSString *> *wsRaw = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:kronosWorkspaceDirectory() error:nil] ?: @[];
        NSUInteger wsCnt = 0;
        for (NSString *f in wsRaw) {
            if (![f hasPrefix:@"."]) wsCnt++;
        }
        [self reindexDocuments];

        NSUInteger cCount = 0;
        CNAuthorizationStatus cnAuth = [CNContactStore authorizationStatusForEntityType:CNEntityTypeContacts];
        CNContactStore *cnStore = [[CNContactStore alloc] init];
        if (cnAuth == CNAuthorizationStatusNotDetermined && promptPermissions) {
            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            [cnStore requestAccessForEntityType:CNEntityTypeContacts completionHandler:^(BOOL granted, NSError *err) {
                (void)granted;
                (void)err;
                dispatch_semaphore_signal(sem);
            }];
            dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(12 * NSEC_PER_SEC)));
            cnAuth = [CNContactStore authorizationStatusForEntityType:CNEntityTypeContacts];
        }
        if (cnAuth == CNAuthorizationStatusAuthorized) {
            CNContactFetchRequest *req = [[CNContactFetchRequest alloc] initWithKeysToFetch:@[CNContactGivenNameKey, CNContactFamilyNameKey]];
            __block NSUInteger acc = 0;
            [cnStore enumerateContactsWithFetchRequest:req error:nil usingBlock:^(CNContact *contact, BOOL *stop) {
                (void)contact;
                (void)stop;
                acc++;
            }];
            cCount = acc;
        }

        NSUInteger pCount = 0;
        NSUInteger todayCnt = 0;
        PHAuthorizationStatus phAuth = [PHPhotoLibrary authorizationStatusForAccessLevel:PHAccessLevelReadWrite];
        if (phAuth == PHAuthorizationStatusNotDetermined && promptPermissions) {
            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            [PHPhotoLibrary requestAuthorizationForAccessLevel:PHAccessLevelReadWrite handler:^(PHAuthorizationStatus status) {
                (void)status;
                dispatch_semaphore_signal(sem);
            }];
            dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(12 * NSEC_PER_SEC)));
            phAuth = [PHPhotoLibrary authorizationStatusForAccessLevel:PHAccessLevelReadWrite];
        }
        if (phAuth == PHAuthorizationStatusAuthorized || phAuth == PHAuthorizationStatusLimited) {
            PHFetchResult<PHAsset *> *allImgs = [PHAsset fetchAssetsWithMediaType:PHAssetMediaTypeImage options:nil];
            pCount = allImgs.count;
            NSCalendar *cal = [NSCalendar currentCalendar];
            NSDate *startOfToday = [cal startOfDayForDate:[NSDate date]];
            PHFetchOptions *todayOpts = [[PHFetchOptions alloc] init];
            todayOpts.predicate = [NSPredicate predicateWithFormat:@"creationDate >= %@", startOfToday];
            PHFetchResult<PHAsset *> *todayImgs = [PHAsset fetchAssetsWithMediaType:PHAssetMediaTypeImage options:todayOpts];
            todayCnt = todayImgs.count;
        }

        NSUInteger calCnt = 0;
        EKAuthorizationStatus ekAuth = [EKEventStore authorizationStatusForEntityType:EKEntityTypeEvent];
        BOOL hasCalAccess = (ekAuth == EKAuthorizationStatusAuthorized);
        if (@available(iOS 17.0, *)) {
            if (ekAuth == EKAuthorizationStatusFullAccess) hasCalAccess = YES;
        }
        if (hasCalAccess) {
            EKEventStore *ek = [[EKEventStore alloc] init];
            NSDate *now = [NSDate date];
            NSDate *week = [now dateByAddingTimeInterval:7 * 86400];
            NSPredicate *pred = [ek predicateForEventsWithStartDate:now endDate:week calendars:nil];
            calCnt = [ek eventsMatchingPredicate:pred].count;
        }

        BOOL macOk = NO;
        NSUInteger macEntries = 0;
        KronosMobileSettings *settings = [KronosMobileSettings sharedSettings];
        if (settings.macSyncEnabled && settings.macServerURL.length > 0) {
            NSString *trimmedBase = [settings.macServerURL stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@"/ "]];
            NSURL *statusURL = [NSURL URLWithString:[trimmedBase stringByAppendingString:@"/v1/device/status"]];
            if (statusURL != nil) {
                NSMutableURLRequest *req = [NSMutableURLRequest requestWithURL:statusURL
                                                                   cachePolicy:NSURLRequestReloadIgnoringLocalCacheData
                                                               timeoutInterval:2.0];
                kronosApplyMacAuth(req);
                dispatch_semaphore_t sem = dispatch_semaphore_create(0);
                __block NSData *respData = nil;
                __block NSInteger statusCode = 0;
                [[[NSURLSession sharedSession] dataTaskWithRequest:req completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
                    if (error == nil && [response isKindOfClass:[NSHTTPURLResponse class]]) {
                        statusCode = [(NSHTTPURLResponse *)response statusCode];
                        respData = data;
                    }
                    dispatch_semaphore_signal(sem);
                }] resume];
                dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(2.3 * NSEC_PER_SEC)));
                if (statusCode == 200 && respData.length > 0) {
                    NSDictionary *json = [NSJSONSerialization JSONObjectWithData:respData options:0 error:nil];
                    if ([json isKindOfClass:[NSDictionary class]]) {
                        macOk = YES;
                        macEntries = (NSUInteger)[json[@"indexed_entries"] integerValue];
                    }
                }
            }
        }

        dispatch_async(dispatch_get_main_queue(), ^{
            self.workspaceFilesCount = wsCnt;
            self.contactsCount = cCount;
            self.photosCount = pCount;
            self.todayPhotosCount = todayCnt;
            self.calendarEventsCount = calCnt;
            self.macReachable = macOk;
            self.macIndexedEntries = macEntries;
            self.lastIndexedAt = [[NSDate date] timeIntervalSince1970];
            self.isRefreshing = NO;
            [self persistCounts];
            if (completion) completion();
        });
    });
}

- (NSString *)buildDeviceContextPromptBlockIfRelevant:(NSString *)query {
    if (![KronosMobileSettings sharedSettings].developDeviceContext) return nil;
    NSString *low = [query ?: @"" lowercaseString];
    NSArray<NSDictionary *> *docHits = [self searchDocuments:query limit:5];
    BOOL asksDevice = [low containsString:@"my phone"] || [low containsString:@"my iphone"] ||
                      [low containsString:@"my device"] || [low containsString:@"workspace"] ||
                      [low containsString:@"my photos"] || [low containsString:@"my contacts"] ||
                      [low containsString:@"photo"] || [low containsString:@"picture"] || [low containsString:@"screenshot"] ||
                      [low containsString:@"access"] || [low containsString:@"permission"] ||
                      [low containsString:@"calendar"] || [low containsString:@"schedule today"] ||
                      [low containsString:@"file"] || [low containsString:@"document"] || docHits.count > 0;
    if (!asksDevice) return nil;
    NSMutableString *docLines = [NSMutableString string];
    for (NSDictionary *d in docHits) {
        [docLines appendFormat:@"\n- Document on this iPhone: %@ (in %@)", d[@"name"], d[@"root"]];
    }

    UIDevice *dev = [UIDevice currentDevice];
    NSArray<NSString *> *wsFiles = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:kronosWorkspaceDirectory() error:nil] ?: @[];
    NSMutableArray<NSString *> *cleanFiles = [NSMutableArray array];
    for (NSString *f in wsFiles) {
        if (![f hasPrefix:@"."]) [cleanFiles addObject:f];
    }
    PHAuthorizationStatus phAuth = [PHPhotoLibrary authorizationStatusForAccessLevel:PHAccessLevelReadWrite];
    NSMutableString *photoLine = [NSMutableString string];
    if (phAuth == PHAuthorizationStatusAuthorized || phAuth == PHAuthorizationStatusLimited) {
        NSDateFormatter *df = [[NSDateFormatter alloc] init];
        df.dateStyle = NSDateFormatterLongStyle;
        df.timeStyle = NSDateFormatterShortStyle;
        NSDate *shot = kronosFetchMedia(1).firstObject.creationDate;
        NSDate *photo = kronosFetchMedia(0).firstObject.creationDate;
        [photoLine appendFormat:@"- Photo Library access: %@. Kronos can read it.\n", phAuth == PHAuthorizationStatusLimited ? @"granted for selected photos" : @"granted"];
        if (shot != nil) [photoLine appendFormat:@"- Latest screenshot taken: %@\n", [df stringFromDate:shot]];
        if (photo != nil) [photoLine appendFormat:@"- Latest photo taken: %@\n", [df stringFromDate:photo]];
    } else {
        [photoLine appendString:@"- Photo Library access: not granted.\n"];
    }
    return [NSString stringWithFormat:
            @"[iPhone Device Context]\n"
             "- Device: %@ (%@ %@)\n"
             "%@"
             "- Indexed on iPhone: %lu contacts, %lu photos (%lu taken today), %lu documents in granted folders, %lu workspace files (%@)%@\n"
             "- Mac Companion Sync: %@\n",
            dev.name, dev.systemName, dev.systemVersion, photoLine,
            (unsigned long)self.contactsCount,
            (unsigned long)self.photosCount,
            (unsigned long)self.todayPhotosCount,
            (unsigned long)self.documentsCount,
            (unsigned long)cleanFiles.count,
            cleanFiles.count > 0 ? [cleanFiles componentsJoinedByString:@", "] : @"none",
            docLines,
            [self macSyncStatusText]];
}

- (NSString *)resolveContactPhoneDigitsForName:(NSString *)targetName outMatchedName:(NSString **)outName {
    NSString *norm = [kronosNormalizeApostrophes(targetName) stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSString *needle = [norm lowercaseString];
    if ([needle hasSuffix:@"'s"]) {
        needle = [needle substringToIndex:needle.length - 2];
    }
    NSSet<NSString *> *badStopTargets = [NSSet setWithArray:@[
        @"", @"on", @"in", @"to", @"from", @"with", @"at", @"for", @"the", @"a", @"an",
        @"me", @"my", @"whatsapp", @"imessage", @"message", @"messages", @"sent", @"last"
    ]];
    if (needle.length < 2 || [badStopTargets containsObject:needle]) return nil;

    CNContactStore *store = [[CNContactStore alloc] init];
    CNAuthorizationStatus auth = [CNContactStore authorizationStatusForEntityType:CNEntityTypeContacts];
    if (auth == CNAuthorizationStatusNotDetermined) {
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        [store requestAccessForEntityType:CNEntityTypeContacts completionHandler:^(BOOL granted, NSError *err) {
            (void)granted;
            (void)err;
            dispatch_semaphore_signal(sem);
        }];
        dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(12 * NSEC_PER_SEC)));
        auth = [CNContactStore authorizationStatusForEntityType:CNEntityTypeContacts];
    }
    if (auth != CNAuthorizationStatusAuthorized) return nil;

    NSString *relRole = nil;
    NSSet<NSString *> *relAliases = kronosRelationshipAliasesForTarget(needle, &relRole);
    NSMutableSet<NSString *> *relatedNamesLower = [NSMutableSet set];

    NSArray *keys = @[
        CNContactGivenNameKey, CNContactMiddleNameKey, CNContactFamilyNameKey,
        CNContactNicknameKey, CNContactOrganizationNameKey,
        CNContactPhoneNumbersKey, CNContactRelationsKey
    ];
    CNContactFetchRequest *req = [[CNContactFetchRequest alloc] initWithKeysToFetch:keys];

    if (relRole.length > 0) {
        [store enumerateContactsWithFetchRequest:req error:nil usingBlock:^(CNContact *c, BOOL *stop) {
            (void)stop;
            for (CNLabeledValue<CNContactRelation *> *rv in c.contactRelations) {
                NSString *lbl = [[CNLabeledValue localizedStringForLabel:(rv.label ?: @"")] lowercaseString];
                NSString *rawLbl = [rv.label ?: @"" lowercaseString];
                if ([lbl containsString:relRole] || [rawLbl containsString:relRole] || [relAliases containsObject:lbl]) {
                    NSString *relPerson = [[rv.value.name ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] lowercaseString];
                    if (relPerson.length >= 2) [relatedNamesLower addObject:relPerson];
                }
            }
        }];
    }

    __block NSString *exactDigits = nil;
    __block NSString *exactLabel = nil;
    __block NSString *wordPrefixDigits = nil;
    __block NSString *wordPrefixLabel = nil;
    NSArray<NSString *> *needleTokens = kronosSplitAlphaWords(needle);

    [store enumerateContactsWithFetchRequest:req error:nil usingBlock:^(CNContact *c, BOOL *stop) {
        if (c.phoneNumbers.count == 0) return;
        NSString *given = c.givenName ?: @"";
        NSString *middle = c.middleName ?: @"";
        NSString *family = c.familyName ?: @"";
        NSString *nick = c.nickname ?: @"";
        NSString *org = c.organizationName ?: @"";
        NSString *full = [[NSString stringWithFormat:@"%@ %@", given, family] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
        NSString *display = full.length > 0 ? full : (nick.length > 0 ? nick : (org.length > 0 ? org : given));

        NSString *gLow = [given lowercaseString];
        NSString *fLow = [family lowercaseString];
        NSString *nLow = [nick lowercaseString];
        NSString *fullLow = [full lowercaseString];

        NSString *rawPhone = c.phoneNumbers.firstObject.value.stringValue ?: @"";
        NSMutableString *digits = [NSMutableString string];
        for (NSUInteger i = 0; i < rawPhone.length; i++) {
            unichar ch = [rawPhone characterAtIndex:i];
            if (ch >= '0' && ch <= '9') {
                [digits appendFormat:@"%C", ch];
            }
        }
        if (digits.length < 7) return;

        BOOL isExact = [gLow isEqualToString:needle] || [fLow isEqualToString:needle] ||
                       [nLow isEqualToString:needle] || [fullLow isEqualToString:needle] ||
                       (relAliases != nil && ([relAliases containsObject:gLow] || [relAliases containsObject:nLow] || [relAliases containsObject:fullLow])) ||
                       (relatedNamesLower.count > 0 && ([relatedNamesLower containsObject:fullLow] || [relatedNamesLower containsObject:gLow]));

        if (isExact) {
            exactDigits = [digits copy];
            exactLabel = display;
            *stop = YES;
            return;
        }

        // Strict word-boundary prefix match (only when needle >= 3 chars, never mid-word substring).
        if (wordPrefixDigits == nil && needle.length >= 3 && needleTokens.count > 0) {
            NSArray<NSString *> *contactWords = kronosSplitAlphaWords([NSString stringWithFormat:@"%@ %@ %@ %@ %@", given, middle, family, nick, org]);
            BOOL allTokensMatchWordStart = YES;
            for (NSString *nt in needleTokens) {
                if (nt.length < 2) continue;
                BOOL tokMatched = NO;
                for (NSString *cw in contactWords) {
                    if ([cw isEqualToString:nt] || (nt.length >= 3 && [cw hasPrefix:nt])) {
                        tokMatched = YES;
                        break;
                    }
                }
                if (!tokMatched) {
                    allTokensMatchWordStart = NO;
                    break;
                }
            }
            if (allTokensMatchWordStart) {
                wordPrefixDigits = [digits copy];
                wordPrefixLabel = display;
            }
        }
    }];

    if (exactDigits != nil) {
        if (outName) *outName = exactLabel;
        return exactDigits;
    }
    if (wordPrefixDigits != nil) {
        if (outName) *outName = wordPrefixLabel;
        return wordPrefixDigits;
    }
    return nil;
}

- (void)lookupContactsForTarget:(NSString *)rawTarget
                     completion:(void (^)(BOOL found, NSString *directReply, NSString *primaryCallDigits))completion {
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        NSString *cleanTarget = [kronosNormalizeApostrophes(rawTarget) stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        if ([[cleanTarget lowercaseString] hasSuffix:@"'s"]) {
            cleanTarget = [cleanTarget substringToIndex:cleanTarget.length - 2];
        }
        NSString *needle = [cleanTarget lowercaseString];

        CNContactStore *store = [[CNContactStore alloc] init];
        CNAuthorizationStatus auth = [CNContactStore authorizationStatusForEntityType:CNEntityTypeContacts];
        if (auth == CNAuthorizationStatusNotDetermined) {
            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            [store requestAccessForEntityType:CNEntityTypeContacts completionHandler:^(BOOL granted, NSError *err) {
                (void)granted;
                (void)err;
                dispatch_semaphore_signal(sem);
            }];
            dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(15 * NSEC_PER_SEC)));
            auth = [CNContactStore authorizationStatusForEntityType:CNEntityTypeContacts];
        }

        if (auth != CNAuthorizationStatusAuthorized) {
            NSString *deniedMsg = @"Contacts access is not enabled for Kronos Mobile yet, so I cannot read your iPhone Contacts (and I will never guess a fake number). Please open iPhone Settings -> Kronos Mobile -> Contacts, enable access, and ask again.";
            dispatch_async(dispatch_get_main_queue(), ^{
                completion(NO, deniedMsg, nil);
            });
            return;
        }

        NSString *relRole = nil;
        NSSet<NSString *> *relAliases = kronosRelationshipAliasesForTarget(needle, &relRole);
        NSMutableSet<NSString *> *relatedNamesLower = [NSMutableSet set];

        NSArray *keys = @[
            CNContactGivenNameKey, CNContactMiddleNameKey, CNContactFamilyNameKey,
            CNContactNicknameKey, CNContactOrganizationNameKey,
            CNContactPhoneNumbersKey, CNContactEmailAddressesKey, CNContactRelationsKey
        ];
        CNContactFetchRequest *req = [[CNContactFetchRequest alloc] initWithKeysToFetch:keys];

        __block NSUInteger totalContacts = 0;
        [store enumerateContactsWithFetchRequest:req error:nil usingBlock:^(CNContact *c, BOOL *stop) {
            (void)stop;
            totalContacts++;
            if (relRole.length > 0) {
                for (CNLabeledValue<CNContactRelation *> *rv in c.contactRelations) {
                    NSString *lbl = [[CNLabeledValue localizedStringForLabel:(rv.label ?: @"")] lowercaseString];
                    NSString *rawLbl = [rv.label ?: @"" lowercaseString];
                    if ([lbl containsString:relRole] || [rawLbl containsString:relRole] || [relAliases containsObject:lbl]) {
                        NSString *relPerson = [[rv.value.name ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] lowercaseString];
                        if (relPerson.length >= 2) [relatedNamesLower addObject:relPerson];
                    }
                }
            }
        }];

        NSArray<NSString *> *needleTokens = kronosSplitAlphaWords(needle);
        NSMutableArray<NSString *> *exactBlocks = [NSMutableArray array];
        NSMutableArray<NSString *> *prefixBlocks = [NSMutableArray array];
        __block NSString *firstDigits = nil;

        [store enumerateContactsWithFetchRequest:req error:nil usingBlock:^(CNContact *c, BOOL *stop) {
            (void)stop;
            NSString *given = c.givenName ?: @"";
            NSString *middle = c.middleName ?: @"";
            NSString *family = c.familyName ?: @"";
            NSString *nick = c.nickname ?: @"";
            NSString *org = c.organizationName ?: @"";
            NSString *full = [[NSString stringWithFormat:@"%@ %@", given, family] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
            NSString *display = full.length > 0 ? full : (nick.length > 0 ? nick : (org.length > 0 ? org : @"Contact"));
            if (nick.length > 0 && full.length > 0 && ![[nick lowercaseString] isEqualToString:[full lowercaseString]]) {
                display = [NSString stringWithFormat:@"%@ (%@)", full, nick];
            }

            NSString *gLow = [given lowercaseString];
            NSString *fLow = [family lowercaseString];
            NSString *nLow = [nick lowercaseString];
            NSString *fullLow = [full lowercaseString];

            BOOL isExact = [gLow isEqualToString:needle] || [fLow isEqualToString:needle] ||
                           [nLow isEqualToString:needle] || [fullLow isEqualToString:needle] ||
                           (relAliases != nil && ([relAliases containsObject:gLow] || [relAliases containsObject:nLow] || [relAliases containsObject:fullLow])) ||
                           (relatedNamesLower.count > 0 && ([relatedNamesLower containsObject:fullLow] || [relatedNamesLower containsObject:gLow]));

            BOOL isPrefix = NO;
            if (!isExact && needle.length >= 3 && needleTokens.count > 0) {
                NSArray<NSString *> *cWords = kronosSplitAlphaWords([NSString stringWithFormat:@"%@ %@ %@ %@ %@", given, middle, family, nick, org]);
                BOOL allOk = YES;
                for (NSString *nt in needleTokens) {
                    if (nt.length < 2) continue;
                    BOOL hit = NO;
                    for (NSString *cw in cWords) {
                        if ([cw isEqualToString:nt] || (nt.length >= 3 && [cw hasPrefix:nt])) {
                            hit = YES;
                            break;
                        }
                    }
                    if (!hit) { allOk = NO; break; }
                }
                isPrefix = allOk;
            }

            if (!isExact && !isPrefix) return;

            NSMutableString *entry = [NSMutableString stringWithFormat:@"%@", display];
            NSMutableArray<NSString *> *lines = [NSMutableArray array];
            for (CNLabeledValue<CNPhoneNumber *> *pv in c.phoneNumbers) {
                NSString *num = pv.value.stringValue ?: @"";
                if (num.length == 0) continue;
                NSString *lbl = pv.label.length > 0 ? [CNLabeledValue localizedStringForLabel:pv.label] : @"phone";
                [lines addObject:[NSString stringWithFormat:@"  • %@: %@", lbl, num]];
                if (firstDigits == nil) {
                    NSMutableString *dg = [NSMutableString string];
                    for (NSUInteger i = 0; i < num.length; i++) {
                        unichar ch = [num characterAtIndex:i];
                        if ((ch >= '0' && ch <= '9') || ch == '+') [dg appendFormat:@"%C", ch];
                    }
                    if (dg.length >= 7) firstDigits = [dg copy];
                }
            }
            for (CNLabeledValue<NSString *> *ev in c.emailAddresses) {
                if (ev.value.length > 0) {
                    NSString *lbl = ev.label.length > 0 ? [CNLabeledValue localizedStringForLabel:ev.label] : @"email";
                    [lines addObject:[NSString stringWithFormat:@"  • %@: %@", lbl, ev.value]];
                }
            }
            if (lines.count == 0) {
                [lines addObject:@"  • (No phone number or email saved on this contact card)"];
            }
            [entry appendFormat:@"\n%@", [lines componentsJoinedByString:@"\n"]];

            if (isExact) {
                [exactBlocks addObject:entry];
            } else if (prefixBlocks.count < 4) {
                [prefixBlocks addObject:entry];
            }
        }];

        NSArray<NSString *> *chosen = exactBlocks.count > 0 ? exactBlocks : prefixBlocks;
        dispatch_async(dispatch_get_main_queue(), ^{
            self.contactsCount = totalContacts;
            [self persistCounts];

            if (chosen.count > 0) {
                NSString *hdr = (chosen.count == 1)
                    ? @"From your iPhone Contacts:\n\n"
                    : [NSString stringWithFormat:@"Found %lu matching entries in your iPhone Contacts:\n\n", (unsigned long)chosen.count];
                NSString *reply = [hdr stringByAppendingString:[chosen componentsJoinedByString:@"\n\n"]];
                completion(YES, reply, firstDigits);
            } else {
                NSString *disp = cleanTarget.length > 0 ? cleanTarget : @"that contact";
                NSString *notFound = nil;
                if (relRole.length > 0) {
                    notFound = [NSString stringWithFormat:
                                @"I checked all %lu contacts on your iPhone and could not find a contact saved as \"%@\" or linked as your %@. I will never invent a fake phone number.\n\nIf your %@ is saved under their real name in Contacts, ask me for their name directly (for example: \"What is <Name>'s phone number?\"), or tell me \"Remember my %@'s phone number is ...\" to store it in Kronos.",
                                (unsigned long)totalContacts, disp, relRole, relRole, [disp lowercaseString]];
                } else {
                    notFound = [NSString stringWithFormat:
                                @"I checked all %lu contacts on your iPhone and could not find a contact matching \"%@\". Please check how their name is spelled in your Contacts app.",
                                (unsigned long)totalContacts, disp];
                }
                completion(NO, notFound, nil);
            }
        });
    });
}

static inline PHFetchResult<PHAsset *> *kronosFetchMedia(NSInteger kind) {
    PHFetchOptions *opts = [[PHFetchOptions alloc] init];
    opts.sortDescriptors = @[[NSSortDescriptor sortDescriptorWithKey:@"creationDate" ascending:NO]];
    if (kind == 2) return [PHAsset fetchAssetsWithMediaType:PHAssetMediaTypeVideo options:opts];
    opts.predicate = [NSPredicate predicateWithFormat:(kind == 1 ? @"(mediaSubtypes & %lu) != 0" : @"(mediaSubtypes & %lu) == 0"),
                      (unsigned long)PHAssetMediaSubtypePhotoScreenshot];
    return [PHAsset fetchAssetsWithMediaType:PHAssetMediaTypeImage options:opts];
}

- (void)fetchLatestMediaOfKind:(NSInteger)kind completion:(void (^)(UIImage *image, NSDate *date, NSUInteger total, BOOL limited, NSString *errorNote))completion {
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        PHAuthorizationStatus auth = [PHPhotoLibrary authorizationStatusForAccessLevel:PHAccessLevelReadWrite];
        if (auth == PHAuthorizationStatusNotDetermined) {
            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            [PHPhotoLibrary requestAuthorizationForAccessLevel:PHAccessLevelReadWrite handler:^(PHAuthorizationStatus status) {
                (void)status;
                dispatch_semaphore_signal(sem);
            }];
            dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(60 * NSEC_PER_SEC)));
            auth = [PHPhotoLibrary authorizationStatusForAccessLevel:PHAccessLevelReadWrite];
        }
        if (auth != PHAuthorizationStatusAuthorized && auth != PHAuthorizationStatusLimited) {
            dispatch_async(dispatch_get_main_queue(), ^{
                completion(nil, nil, 0, NO, @"Photo Library access is off for Kronos. Turn it on in iPhone Settings, then ask again.");
            });
            return;
        }
        BOOL limited = (auth == PHAuthorizationStatusLimited);
        PHFetchResult<PHAsset *> *assets = kronosFetchMedia(kind);
        PHAsset *latest = assets.firstObject;
        NSUInteger total = assets.count;
        __block UIImage *image = nil;
        if (latest != nil) {
            PHImageRequestOptions *o = [[PHImageRequestOptions alloc] init];
            o.synchronous = YES;
            o.deliveryMode = PHImageRequestOptionsDeliveryModeHighQualityFormat;
            o.resizeMode = PHImageRequestOptionsResizeModeFast;
            o.networkAccessAllowed = YES;
            [[PHImageManager defaultManager] requestImageForAsset:latest targetSize:CGSizeMake(1440.0, 1440.0)
                                                      contentMode:PHImageContentModeAspectFit options:o
                                                    resultHandler:^(UIImage *result, NSDictionary *info) {
                (void)info;
                if (result != nil) image = result;
            }];
        }
        NSDate *date = latest.creationDate;
        dispatch_async(dispatch_get_main_queue(), ^{ completion(image, date, total, limited, nil); });
    });
}

- (void)fetchLatestPhotoForPostWithCompletion:(void (^)(UIImage *image, NSDate *takenDate, BOOL takenToday, NSString *errorNote))completion {
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        PHAuthorizationStatus auth = [PHPhotoLibrary authorizationStatusForAccessLevel:PHAccessLevelReadWrite];
        if (auth == PHAuthorizationStatusNotDetermined) {
            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            [PHPhotoLibrary requestAuthorizationForAccessLevel:PHAccessLevelReadWrite handler:^(PHAuthorizationStatus status) {
                (void)status;
                dispatch_semaphore_signal(sem);
            }];
            dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(15 * NSEC_PER_SEC)));
            auth = [PHPhotoLibrary authorizationStatusForAccessLevel:PHAccessLevelReadWrite];
        }
        if (auth != PHAuthorizationStatusAuthorized && auth != PHAuthorizationStatusLimited) {
            dispatch_async(dispatch_get_main_queue(), ^{
                completion(nil, nil, NO, @"Photo Library access is not enabled yet. Please grant Photo Library permission in iPhone Settings -> Kronos Mobile so Kronos can pick the last picture you took today.");
            });
            return;
        }

        PHFetchOptions *opts = [[PHFetchOptions alloc] init];
        opts.sortDescriptors = @[[NSSortDescriptor sortDescriptorWithKey:@"creationDate" ascending:NO]];
        opts.fetchLimit = 10;
        PHFetchResult<PHAsset *> *assets = [PHAsset fetchAssetsWithMediaType:PHAssetMediaTypeImage options:opts];
        if (assets.count == 0) {
            dispatch_async(dispatch_get_main_queue(), ^{
                completion(nil, nil, NO, @"No photos were found in your iPhone Photo Library.");
            });
            return;
        }

        NSCalendar *cal = [NSCalendar currentCalendar];
        PHAsset *chosen = nil;
        BOOL isToday = NO;
        for (PHAsset *a in assets) {
            if (a.creationDate != nil && [cal isDateInToday:a.creationDate]) {
                chosen = a;
                isToday = YES;
                break;
            }
        }
        if (chosen == nil) {
            chosen = assets.firstObject;
            isToday = (chosen.creationDate != nil && [cal isDateInToday:chosen.creationDate]);
        }

        PHImageRequestOptions *imgOpts = [[PHImageRequestOptions alloc] init];
        imgOpts.synchronous = YES;
        imgOpts.deliveryMode = PHImageRequestOptionsDeliveryModeHighQualityFormat;
        imgOpts.resizeMode = PHImageRequestOptionsResizeModeExact;
        imgOpts.networkAccessAllowed = YES;

        __block UIImage *fetched = nil;
        [[PHImageManager defaultManager] requestImageForAsset:chosen
                                                   targetSize:CGSizeMake(1440.0, 1440.0)
                                                  contentMode:PHImageContentModeAspectFit
                                                      options:imgOpts
                                                resultHandler:^(UIImage *result, NSDictionary *info) {
            (void)info;
            if (result != nil) fetched = result;
        }];

        NSDate *cDate = chosen.creationDate ?: [NSDate date];
        dispatch_async(dispatch_get_main_queue(), ^{
            if (fetched == nil) {
                completion(nil, nil, NO, @"Could not load the latest photo from your Photo Library.");
            } else {
                completion(fetched, cDate, isToday, nil);
            }
        });
    });
}
@end

static inline NSAttributedString *kronosFormatMessageText(NSString *raw, BOOL isUserBubble) {
    UIFont *bodyFont = [UIFont systemFontOfSize:15.5 weight:UIFontWeightRegular];
    UIFont *codeFont = [UIFont monospacedSystemFontOfSize:13.5 weight:UIFontWeightRegular];
    UIColor *bodyColor = isUserBubble ? [UIColor whiteColor] : kronosTextColor();
    UIColor *codeColor = isUserBubble ? [UIColor colorWithWhite:1.0 alpha:0.95] : kronosAccentColor();

    NSMutableParagraphStyle *para = [[NSMutableParagraphStyle alloc] init];
    para.lineSpacing = 3.5;

    NSMutableAttributedString *out = [[NSMutableAttributedString alloc] init];
    NSArray<NSString *> *parts = [(raw ?: @"") componentsSeparatedByString:@"```"];
    for (NSUInteger i = 0; i < parts.count; i++) {
        NSString *chunk = parts[i];
        BOOL isCode = (i % 2 == 1);
        if (isCode) {
            NSRange nl = [chunk rangeOfString:@"\n"];
            if (nl.location != NSNotFound && nl.location < 20) {
                chunk = [chunk substringFromIndex:nl.location + 1];
            }
            chunk = [chunk stringByTrimmingCharactersInSet:[NSCharacterSet newlineCharacterSet]];
            if (out.length > 0) {
                [out appendAttributedString:[[NSAttributedString alloc] initWithString:@"\n"]];
            }
            NSAttributedString *attr = [[NSAttributedString alloc] initWithString:chunk
                                                                       attributes:@{
                NSFontAttributeName: codeFont,
                NSForegroundColorAttributeName: codeColor,
                NSParagraphStyleAttributeName: para,
            }];
            [out appendAttributedString:attr];
            if (i + 1 < parts.count) {
                [out appendAttributedString:[[NSAttributedString alloc] initWithString:@"\n"]];
            }
        } else if (chunk.length > 0) {
            NSAttributedString *attr = [[NSAttributedString alloc] initWithString:chunk
                                                                       attributes:@{
                NSFontAttributeName: bodyFont,
                NSForegroundColorAttributeName: bodyColor,
                NSParagraphStyleAttributeName: para,
            }];
            [out appendAttributedString:attr];
        }
    }
    return out;
}

/// True while a reply has an unterminated `<think>` block (the model is still reasoning).
static inline BOOL kronosHasOpenThinkBlock(NSString *text) {
    NSRange open = [text rangeOfString:@"<think>" options:NSBackwardsSearch];
    if (open.location == NSNotFound) return NO;
    NSRange close = [text rangeOfString:@"</think>" options:NSBackwardsSearch];
    return close.location == NSNotFound || close.location < open.location;
}

/// Removes every `<think>…</think>` block (and a trailing unterminated one) from a reply.
static inline NSString *kronosStripThinkBlocks(NSString *text) {
    if (text.length == 0 || [text rangeOfString:@"<think>"].location == NSNotFound) return text ?: @"";
    NSMutableString *out = [NSMutableString string];
    NSUInteger pos = 0;
    while (pos < text.length) {
        NSRange open = [text rangeOfString:@"<think>" options:0 range:NSMakeRange(pos, text.length - pos)];
        if (open.location == NSNotFound) {
            [out appendString:[text substringFromIndex:pos]];
            break;
        }
        [out appendString:[text substringWithRange:NSMakeRange(pos, open.location - pos)]];
        NSUInteger after = open.location + open.length;
        NSRange close = [text rangeOfString:@"</think>" options:0 range:NSMakeRange(after, text.length - after)];
        if (close.location == NSNotFound) break;
        pos = close.location + close.length;
    }
    return [out stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
}

/// "find / open / where is / show me / do I have … my X (on this phone)" -> X; nil for app, chat or
/// messaging requests.
static inline BOOL kronosParseFindFileIntent(NSString *query, NSString **outTerm) {
    NSString *low = [[query ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] lowercaseString];
    low = [low stringByReplacingOccurrencesOfString:@"?" withString:@""];
    low = [low stringByReplacingOccurrencesOfString:@"." withString:@""];
    NSArray<NSString *> *lead = @[@"do i have access to ", @"do i have ", @"can you find ", @"please find ", @"find ", @"open ", @"where is ", @"where's ", @"show me ", @"locate ", @"look for ", @"search for ", @"get me "];
    NSString *term = nil;
    for (NSString *l in lead) {
        if ([low hasPrefix:l]) { term = [low substringFromIndex:l.length]; break; }
    }
    if (term == nil) return NO;
    NSArray<NSString *> *tails = @[@" on this phone", @" on my phone", @" on this iphone", @" on my iphone", @" on this device", @" on my device", @" in this phone", @" in my phone", @" in this iphone", @" in my iphone", @" here", @" for me", @" please"];
    BOOL changed = YES;
    while (changed) {
        changed = NO;
        for (NSString *t in tails) {
            if ([term hasSuffix:t]) { term = [term substringToIndex:term.length - t.length]; changed = YES; }
        }
        term = [term stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    }
    NSRange findIt = [term rangeOfString:@" find it"];
    if (findIt.location != NSNotFound) term = [term substringToIndex:findIt.location];
    for (NSString *pre in @[@"my ", @"the ", @"a ", @"an "]) {
        if ([term hasPrefix:pre]) { term = [term substringFromIndex:pre.length]; break; }
    }
    term = [term stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (term.length < 2 || [term componentsSeparatedByString:@" "].count > 6) return NO;
    if ([@[@"my", @"the", @"it", @"this", @"that"] containsObject:term]) return NO;
    // Keep the user's spelling ("ND result") when the lowercase mapping preserved offsets.
    NSString *orig = [query stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSString *origLow = [[orig stringByReplacingOccurrencesOfString:@"?" withString:@""] stringByReplacingOccurrencesOfString:@"." withString:@""];
    NSRange r = [[origLow lowercaseString] rangeOfString:term];
    if (r.location != NSNotFound && origLow.length == [origLow lowercaseString].length) {
        term = [origLow substringWithRange:r];
    }
    for (NSString *app in @[@"whatsapp", @"safari", @"github", @"linkedin", @"settings", @"camera", @"photos", @"mail", @"messages", @"maps", @"calendar", @"notes", @"music", @"spotify", @"youtube", @"instagram", @"chrome", @"app "]) {
        if ([term hasPrefix:app]) return NO;
    }
    if (outTerm != NULL) *outTerm = term;
    return YES;
}

typedef NS_ENUM(NSInteger, KronosAppAsk) { KronosAppAskHas, KronosAppAskCount, KronosAppAskOpen };

/// Name matches from a word boundary, ignoring spaces and doubled letters ("Uloo" finds "Ulo").
static inline BOOL kronosAppNameMatches(NSString *name, NSString *query) {
    NSString *(^squash)(NSString *) = ^NSString *(NSString *t) {
        NSMutableString *out = [NSMutableString string];
        unichar prev = 0;
        NSString *low = [t lowercaseString];
        for (NSUInteger i = 0; i < low.length; i++) {
            unichar c = [low characterAtIndex:i];
            if (![[NSCharacterSet alphanumericCharacterSet] characterIsMember:c] || c == prev) continue;
            [out appendFormat:@"%C", c];
            prev = c;
        }
        return out;
    };
    NSString *want = squash(query);
    if (want.length == 0) return NO;
    NSArray<NSString *> *words = [name componentsSeparatedByCharactersInSet:[[NSCharacterSet alphanumericCharacterSet] invertedSet]];
    for (NSUInteger i = 0; i < words.count; i++) {
        NSString *tail = [[words subarrayWithRange:NSMakeRange(i, words.count - i)] componentsJoinedByString:@" "];
        if ([squash(tail) hasPrefix:want]) return YES;
    }
    return NO;
}

/// App questions beyond "do I have": "how many Ulo apps", "open the Justa app", "how about the X app".
static inline BOOL kronosParseAppAsk(NSString *query, BOOL afterAppTurn, NSString **outApp, KronosAppAsk *outAsk) {
    NSString *q = [[kronosNormalizeApostrophes(query ?: @"") stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@"?. "]];
    NSString *low = [q lowercaseString];
    NSString *rest = nil;
    KronosAppAsk ask = KronosAppAskHas;
    if ([low hasPrefix:@"how many "]) {
        rest = [q substringFromIndex:9];
        ask = KronosAppAskCount;
    } else {
        for (NSString *lead in @[@"how about ", @"what about ", @"and "]) {
            if ([low hasPrefix:lead]) { rest = [q substringFromIndex:lead.length]; break; }
        }
        if (rest == nil) {
            for (NSString *lead in @[@"open ", @"launch ", @"start "]) {
                if ([low hasPrefix:lead]) { rest = [q substringFromIndex:lead.length]; ask = KronosAppAskOpen; break; }
            }
        }
    }
    if (rest == nil) return NO;
    NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:@"(\\s+(do|did|are|have|on)\\b.*)$" options:NSRegularExpressionCaseInsensitive error:nil];
    rest = [re stringByReplacingMatchesInString:rest options:0 range:NSMakeRange(0, rest.length) withTemplate:@""];
    BOOL sawApp = NO;
    NSString *trimmedLow = [rest lowercaseString];
    if ([trimmedLow isEqualToString:@"apps"] || [trimmedLow isEqualToString:@"app"]) { rest = @""; sawApp = YES; }
    for (NSString *t in @[@" apps", @" app", @" applications", @" application"]) {
        if ([[rest lowercaseString] hasSuffix:t]) { rest = [rest substringToIndex:rest.length - t.length]; sawApp = YES; break; }
    }
    if (!sawApp && !afterAppTurn) return NO;
    for (NSString *pre in @[@"the ", @"my ", @"an ", @"a "]) {
        if ([[rest lowercaseString] hasPrefix:pre]) { rest = [rest substringFromIndex:pre.length]; break; }
    }
    rest = [rest stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
    if (rest.length == 0 && ask != KronosAppAskCount) return NO;
    if ([rest componentsSeparatedByString:@" "].count > 4) return NO;
    if (outApp != NULL) *outApp = rest;
    if (outAsk != NULL) *outAsk = ask;
    return YES;
}

/// "Do I have the Justa app on my phone?", "Is Spotify installed?", "Do I have WhatsApp?".
static inline BOOL kronosParseHasAppIntent(NSString *query, NSString **outApp) {
    NSString *q = [[kronosNormalizeApostrophes(query ?: @"") stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@"?. "]];
    NSString *low = [q lowercaseString];
    NSString *rest = nil;
    for (NSString *lead in @[@"do i have ", @"have i got ", @"have i installed ", @"did i install ", @"is ", @"is there "]) {
        if ([low hasPrefix:lead]) { rest = [q substringFromIndex:lead.length]; }
    }
    if (rest == nil) return NO;
    BOOL viaIs = [low hasPrefix:@"is "];
    NSArray<NSString *> *tails = @[@" on my phone", @" on this phone", @" on my iphone", @" on this iphone", @" on my device", @" on here", @" on it", @" installed", @" downloaded"];
    BOOL sawTail = NO, sawInstalled = NO;
    for (BOOL changed = YES; changed;) {
        changed = NO;
        for (NSString *t in tails) {
            if ([[rest lowercaseString] hasSuffix:t]) {
                rest = [rest substringToIndex:rest.length - t.length];
                sawTail = YES;
                if ([t isEqualToString:@" installed"] || [t isEqualToString:@" downloaded"]) sawInstalled = YES;
                changed = YES;
            }
        }
    }
    BOOL sawApp = NO;
    for (NSString *t in @[@" app", @" application"]) {
        if ([[rest lowercaseString] hasSuffix:t]) { rest = [rest substringToIndex:rest.length - t.length]; sawApp = YES; }
    }
    for (NSString *pre in @[@"the ", @"an ", @"a "]) {
        if ([[rest lowercaseString] hasPrefix:pre]) { rest = [rest substringFromIndex:pre.length]; break; }
    }
    rest = [rest stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
    if (rest.length < 2 || [rest componentsSeparatedByString:@" "].count > 4) return NO;
    NSArray *known = @[@"whatsapp", @"instagram", @"telegram", @"slack", @"youtube", @"twitter", @"x", @"github", @"linkedin", @"gmail", @"spotify", @"tiktok", @"facebook", @"snapchat", @"chrome", @"netflix", @"uber", @"bolt", @"zoom"];
    BOOL names = sawApp || sawInstalled || (!viaIs && sawTail && [known containsObject:[rest lowercaseString]]) || (!viaIs && [known containsObject:[rest lowercaseString]]);
    if (!names) return NO;
    if (outApp != NULL) *outApp = rest;
    return YES;
}

/// "On my Mac, fix the failing tests" or "fix the failing tests on my Mac".
static inline BOOL kronosParseMacTaskIntent(NSString *query, NSString **outTask) {
    NSString *q = [query ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSString *low = [q lowercaseString];
    NSString *task = nil;
    for (NSString *lead in @[@"start on my mac", @"run on my mac", @"on my mac", @"on the mac", @"tell my mac to", @"ask my mac to", @"have my mac", @"get my mac to"]) {
        if ([low hasPrefix:lead]) { task = [q substringFromIndex:lead.length]; break; }
    }
    if (task == nil) {
        for (NSString *tail in @[@" on my mac", @" on the mac", @" on my macbook"]) {
            NSString *cut = [low hasSuffix:@"."] ? [low substringToIndex:low.length - 1] : low;
            if ([cut hasSuffix:tail]) { task = [q substringToIndex:cut.length - tail.length]; break; }
        }
        if (task != nil) {
            NSString *first = [[[task lowercaseString] componentsSeparatedByString:@" "] firstObject];
            NSArray *asks = @[@"what", @"what's", @"whats", @"where", @"who", @"when", @"why", @"how", @"is", @"are", @"do", @"does", @"did", @"was", @"has", @"have"];
            if ([asks containsObject:first]) return NO;
        }
    }
    if (task == nil) return NO;
    NSMutableCharacterSet *trim = [NSMutableCharacterSet whitespaceAndNewlineCharacterSet];
    [trim addCharactersInString:@",:;-"];
    task = [task stringByTrimmingCharactersInSet:trim];
    for (NSString *pre in @[@"can you ", @"could you ", @"please "]) {
        if ([[task lowercaseString] hasPrefix:pre]) task = [task substringFromIndex:pre.length];
    }
    if (task.length < 3) return NO;
    if (outTask != NULL) *outTask = task;
    return YES;
}

static inline BOOL kronosIsMacTaskStatusQuery(NSString *query) {
    NSString *low = [[query ?: @"" lowercaseString] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    for (NSString *c in @[@"what is my mac doing", @"what's my mac doing", @"whats my mac doing", @"is my mac done", @"is my mac finished", @"mac task status", @"my mac task", @"task on my mac"]) {
        if ([low containsString:c]) return YES;
    }
    return NO;
}

/// "last screenshot I took", "when was my latest photo", "how many videos do I have".
/// kind: 0 photo, 1 screenshot, 2 video.
static inline BOOL kronosParseLatestMediaIntent(NSString *query, NSInteger *outKind, BOOL *outCount) {
    NSString *low = [[query ?: @"" lowercaseString] stringByReplacingOccurrencesOfString:@"screen shot" withString:@"screenshot"];
    NSCharacterSet *sep = [[NSCharacterSet alphanumericCharacterSet] invertedSet];
    NSMutableArray<NSString *> *words = [[low componentsSeparatedByCharactersInSet:sep] mutableCopy];
    [words removeObject:@""];
    if (words.count == 0 || words.count > 16) return NO;
    NSSet<NSString *> *set = [NSSet setWithArray:words];
    NSInteger kind = -1;
    if ([set containsObject:@"screenshot"] || [set containsObject:@"screenshots"]) kind = 1;
    else if ([set containsObject:@"video"] || [set containsObject:@"videos"]) kind = 2;
    else if ([set intersectsSet:[NSSet setWithArray:@[@"photo", @"photos", @"picture", @"pictures", @"pic", @"pics", @"image", @"images", @"selfie"]]]) kind = 0;
    if (kind < 0) return NO;
    BOOL personal = [set intersectsSet:[NSSet setWithArray:@[@"my", @"i", @"me"]]] || [words.firstObject isEqualToString:@"show"] || [words.firstObject isEqualToString:@"open"];
    if (!personal) return NO;
    BOOL count = [low containsString:@"how many"];
    BOOL recent = [set intersectsSet:[NSSet setWithArray:@[@"last", @"latest", @"newest", @"recent"]]];
    if (!count && !recent) return NO;
    if (outKind != NULL) *outKind = kind;
    if (outCount != NULL) *outCount = count;
    return YES;
}

/// Files whose name matches every word of `term`: words of 3+ letters as substrings, shorter ones
/// as whole words, so "nd result" does not match "index.html".
static inline BOOL kronosFileNameMatchesTerm(NSString *filename, NSString *term) {
    NSString *name = [filename lowercaseString];
    NSCharacterSet *sep = [[NSCharacterSet alphanumericCharacterSet] invertedSet];
    NSSet<NSString *> *nameWords = [NSSet setWithArray:[name componentsSeparatedByCharactersInSet:sep]];
    NSSet<NSString *> *stop = [NSSet setWithArray:@[@"result", @"results", @"file", @"files", @"document", @"documents", @"doc", @"copy", @"pdf", @"image", @"photo", @"picture", @"screenshot", @"of", @"for", @"and", @"my"]];
    NSUInteger checked = 0;
    for (NSString *w in [[term lowercaseString] componentsSeparatedByCharactersInSet:sep]) {
        if (w.length == 0 || [stop containsObject:w]) continue;
        checked++;
        if (w.length >= 3) {
            if ([name rangeOfString:w].location == NSNotFound) return NO;
        } else if (![nameWords containsObject:w]) {
            return NO;
        }
    }
    return checked > 0;
}

static inline NSString *kronosSanitizeFirstPersonReply(NSString *reply) {
    if (reply.length == 0) return reply;
    NSString *fixed = [reply stringByReplacingOccurrencesOfString:@"Melody Leonard built you"
                                                       withString:@"Melody Leonard built me"];
    fixed = [fixed stringByReplacingOccurrencesOfString:@"Melody Leonard Oparaugo built you"
                                             withString:@"Melody Leonard Oparaugo built me"];
    fixed = [fixed stringByReplacingOccurrencesOfString:@"Melody Leonard created you"
                                             withString:@"Melody Leonard created me"];
    fixed = [fixed stringByReplacingOccurrencesOfString:@"You were built by Melody Leonard"
                                             withString:@"I was built by Melody Leonard"];
    fixed = [fixed stringByReplacingOccurrencesOfString:@"You were created by Melody Leonard"
                                             withString:@"I was created by Melody Leonard"];
    fixed = [fixed stringByReplacingOccurrencesOfString:@"created you on the Kronos"
                                             withString:@"created me on the Kronos"];
    NSString *trimmed = [fixed stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSString *low = [trimmed lowercaseString];
    if ([low isEqualToString:@"i created kronos"] || [low isEqualToString:@"i created kronos."] ||
        [low isEqualToString:@"i made kronos"] || [low isEqualToString:@"i made kronos."] ||
        [low isEqualToString:@"i built kronos"] || [low isEqualToString:@"i built kronos."]) {
        return @"Melody Leonard created and built me (Kronos). He is a Senior Software Engineer and Systems Architect with over 10 years of software engineering experience (@IPMScoutek).";
    }
    if ([low isEqualToString:@"i am melody leonard"] || [low isEqualToString:@"i am melody leonard."] ||
        [low isEqualToString:@"i'm melody leonard"] || [low isEqualToString:@"i'm melody leonard."]) {
        return @"Melody Leonard (Melody Leonard Oparaugo) is a Senior Software Engineer and Systems Architect with over 10 years of software engineering experience (@IPMScoutek), and the creator of Kronos.";
    }
    return fixed;
}

@interface KronosAttachmentItem : NSObject
@property (nonatomic, copy) NSString *itemId;
@property (nonatomic, copy) NSString *filename;
@property (nonatomic, copy) NSString *mimeType;
@property (nonatomic, strong) NSData *rawData;
@property (nonatomic, strong) UIImage *thumbnail;
@property (nonatomic, copy) NSString *summaryBadge;
@property (nonatomic, copy) NSString *promptContext;
@property (nonatomic, copy) NSString *workspacePath;
@property (nonatomic, assign) BOOL isImage;
@end

@implementation KronosAttachmentItem
@end

// Always normalizes UIImage EXIF rotation into an upright 1.0-scale CGImage bitmap.
static inline UIImage *kronosRenderUprightImage(UIImage *img, CGFloat maxDim) {
    if (img == nil) return nil;
    CGFloat srcW = MAX(img.size.width, 1.0);
    CGFloat srcH = MAX(img.size.height, 1.0);
    CGFloat scale = MIN(maxDim / srcW, maxDim / srcH);
    scale = MIN(scale, 1.0);
    CGSize targetSize = CGSizeMake(MAX(round(srcW * scale), 1.0), MAX(round(srcH * scale), 1.0));
    UIGraphicsImageRendererFormat *fmt = [UIGraphicsImageRendererFormat defaultFormat];
    fmt.scale = 1.0;
    fmt.opaque = NO;
    UIGraphicsImageRenderer *renderer = [[UIGraphicsImageRenderer alloc] initWithSize:targetSize format:fmt];
    return [renderer imageWithActions:^(UIGraphicsImageRendererContext *ctx) {
        (void)ctx;
        [img drawInRect:CGRectMake(0, 0, targetSize.width, targetSize.height)];
    }];
}

static inline NSString *kronosSampleSurfaceToneSummary(CGImageRef cgImg) {
    if (cgImg == NULL) return @"natural lighting";
    size_t sw = 24, sh = 24;
    uint8_t rgba[24 * 24 * 4] = {0};
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(rgba, sw, sh, 8, sw * 4, cs,
                                             kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(cs);
    if (ctx == NULL) return @"natural lighting";
    CGContextDrawImage(ctx, CGRectMake(0, 0, sw, sh), cgImg);
    CGContextRelease(ctx);

    NSUInteger silverGray = 0, warmWoodBrown = 0, darkTone = 0, blueTone = 0, greenTone = 0;
    double lumAcc = 0.0;
    NSUInteger total = sw * sh;
    for (size_t i = 0; i < total; i++) {
        int r = rgba[i * 4 + 0];
        int g = rgba[i * 4 + 1];
        int b = rgba[i * 4 + 2];
        int maxC = MAX(r, MAX(g, b));
        int minC = MIN(r, MIN(g, b));
        int spread = maxC - minC;
        lumAcc += (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
        if (maxC < 55) {
            darkTone++;
        } else if (spread < 22 && r >= 115 && r <= 235) {
            silverGray++;
        } else if (r > g + 18 && g > b + 12 && r >= 70 && r <= 195) {
            warmWoodBrown++;
        } else if (b > r + 25 && b > g + 12) {
            blueTone++;
        } else if (g > r + 18 && g > b + 18) {
            greenTone++;
        }
    }
    NSMutableArray<NSString *> *tones = [NSMutableArray array];
    if (silverGray * 100 / total >= 16) [tones addObject:@"silver / metallic gray"];
    if (warmWoodBrown * 100 / total >= 14) [tones addObject:@"warm brown wood tone"];
    if (darkTone * 100 / total >= 24) [tones addObject:@"matte black / dark trim"];
    if (blueTone * 100 / total >= 16) [tones addObject:@"blue"];
    if (greenTone * 100 / total >= 16) [tones addObject:@"green"];
    double avgLum = lumAcc / (double)total;
    NSString *lightDesc = (avgLum > 0.62) ? @"bright lighting" : ((avgLum < 0.30) ? @"low-key dark lighting" : @"balanced indoor lighting");
    if (tones.count > 0) {
        return [NSString stringWithFormat:@"%@ (%@)", [tones componentsJoinedByString:@", "], lightDesc];
    }
    return lightDesc;
}

static inline NSString *kronosFriendlyVisionObjectName(NSString *rawId) {
    static NSDictionary<NSString *, NSString *> *map = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        map = @{
            @"laptop": @"laptop computer (notebook / MacBook)",
            @"portable_computer": @"laptop computer (notebook / MacBook)",
            @"notebook_computer": @"laptop computer",
            @"computer": @"computer",
            @"desktop_computer": @"desktop computer",
            @"computer_keyboard": @"computer keyboard",
            @"keyboard": @"keyboard",
            @"computer_mouse": @"computer mouse",
            @"monitor": @"display monitor",
            @"display": @"screen / display",
            @"television": @"television / monitor",
            @"cellular_telephone": @"smartphone",
            @"smartphone": @"smartphone",
            @"mobile_phone": @"mobile phone",
            @"tablet_computer": @"tablet",
            @"wood_processed": @"wooden desk / wood surface",
            @"hardwood": @"hardwood surface",
            @"wood": @"wooden surface",
            @"table": @"desk / table",
            @"desk": @"desk",
            @"cord": @"charging / power cable",
            @"cable": @"cable",
            @"headphones": @"headphones",
            @"earphones": @"earbuds / earphones",
            @"coffee_cup": @"coffee cup / mug",
            @"mug": @"mug",
            @"bottle": @"bottle",
            @"water_bottle": @"water bottle",
            @"watch": @"wristwatch",
            @"wristwatch": @"wristwatch",
            @"eyeglasses": @"glasses",
            @"sunglasses": @"sunglasses",
            @"backpack": @"backpack",
            @"book": @"book",
            @"notebook": @"notebook",
            @"car": @"car",
            @"automobile": @"car",
        };
    });
    NSString *mapped = map[rawId];
    if (mapped != nil) return mapped;
    return [rawId stringByReplacingOccurrencesOfString:@"_" withString:@" "];
}

static inline BOOL kronosIsAbstractVisionHypernym(NSString *rawId, BOOL suppressDocTags) {
    static NSSet<NSString *> *hypernyms = nil;
    static NSSet<NSString *> *docTags = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        hypernyms = [NSSet setWithArray:@[
            @"machine", @"consumer_electronics", @"electronics", @"electronic_device",
            @"device", @"equipment", @"apparatus", @"artifact", @"object", @"structure",
            @"material", @"matter", @"solid", @"surface", @"furniture", @"furnishing",
            @"appliance", @"implement", @"tool", @"utensil", @"container", @"covering",
            @"indoor", @"outdoor", @"interior", @"room", @"building_material", @"product",
            @"merchandise", @"whole", @"part", @"component"
        ]];
        docTags = [NSSet setWithArray:@[
            @"document", @"printed_page", @"screenshot", @"text", @"sign", @"label", @"paper"
        ]];
    });
    if ([hypernyms containsObject:rawId]) return YES;
    if (suppressDocTags && [docTags containsObject:rawId]) return YES;
    return NO;
}

static inline NSDictionary *kronosAnalyzeUprightImageWithVision(UIImage *upright) {
    if (upright == nil || upright.CGImage == nil) {
        return @{@"objects": @[], @"ocrLines": @[], @"ocrChars": @(0), @"faces": @(0), @"humans": @(0), @"tones": @""};
    }
    CGImageRef cg = upright.CGImage;

    // 1. OCR pass (run independently so Vision requests never fail as a coupled batch).
    NSMutableArray<NSString *> *ocrLines = [NSMutableArray array];
    __block NSUInteger totalOcrChars = 0;
    VNRecognizeTextRequest *ocrReq = [[VNRecognizeTextRequest alloc] init];
    ocrReq.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
    ocrReq.usesLanguageCorrection = YES;
    VNImageRequestHandler *ocrHandler = [[VNImageRequestHandler alloc] initWithCGImage:cg
                                                                           orientation:kCGImagePropertyOrientationUp
                                                                               options:@{}];
    [ocrHandler performRequests:@[ocrReq] error:nil];
    if (ocrReq.results.count > 0) {
        NSArray<VNRecognizedTextObservation *> *sorted = [ocrReq.results sortedArrayUsingComparator:^NSComparisonResult(VNRecognizedTextObservation *a, VNRecognizedTextObservation *b) {
            CGFloat ay = a.boundingBox.origin.y;
            CGFloat by = b.boundingBox.origin.y;
            if (fabs(ay - by) > 0.025) {
                return (ay > by) ? NSOrderedAscending : NSOrderedDescending;
            }
            return (a.boundingBox.origin.x < b.boundingBox.origin.x) ? NSOrderedAscending : NSOrderedDescending;
        }];
        for (VNRecognizedTextObservation *o in sorted) {
            VNRecognizedText *top = [[o topCandidates:1] firstObject];
            if (top != nil && top.confidence >= 0.45) {
                NSString *trimmed = [top.string stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
                if (trimmed.length > 0) {
                    [ocrLines addObject:trimmed];
                    for (NSUInteger i = 0; i < trimmed.length; i++) {
                        unichar ch = [trimmed characterAtIndex:i];
                        if ([[NSCharacterSet alphanumericCharacterSet] characterIsMember:ch]) {
                            totalOcrChars++;
                        }
                    }
                }
            }
        }
    }

    BOOL isPhysicalPhoto = (totalOcrChars < 35);

    // 2. Scene & concrete object classification pass (sorted explicitly by confidence descending).
    NSMutableArray<NSString *> *concreteObjects = [NSMutableArray array];
    NSMutableSet<NSString *> *seenFriendly = [NSMutableSet set];
    VNClassifyImageRequest *clsReq = [[VNClassifyImageRequest alloc] init];
    VNImageRequestHandler *clsHandler = [[VNImageRequestHandler alloc] initWithCGImage:cg
                                                                           orientation:kCGImagePropertyOrientationUp
                                                                               options:@{}];
    [clsHandler performRequests:@[clsReq] error:nil];
    if (clsReq.results.count > 0) {
        NSArray<VNClassificationObservation *> *sortedCls = [clsReq.results sortedArrayUsingComparator:^NSComparisonResult(VNClassificationObservation *a, VNClassificationObservation *b) {
            if (a.confidence > b.confidence) return NSOrderedAscending;
            if (a.confidence < b.confidence) return NSOrderedDescending;
            return NSOrderedSame;
        }];
        for (VNClassificationObservation *c in sortedCls) {
            if (c.confidence < 0.06) break;
            if (kronosIsAbstractVisionHypernym(c.identifier, isPhysicalPhoto)) continue;
            NSString *friendly = kronosFriendlyVisionObjectName(c.identifier);
            if ([seenFriendly containsObject:friendly]) continue;
            [seenFriendly addObject:friendly];
            [concreteObjects addObject:[NSString stringWithFormat:@"%@ (%.0f%%)", friendly, c.confidence * 100.0]];
            if (concreteObjects.count >= 7) break;
        }
    }

    // 3. Animal detection pass.
    VNRecognizeAnimalsRequest *animReq = [[VNRecognizeAnimalsRequest alloc] init];
    VNImageRequestHandler *animHandler = [[VNImageRequestHandler alloc] initWithCGImage:cg
                                                                            orientation:kCGImagePropertyOrientationUp
                                                                                options:@{}];
    [animHandler performRequests:@[animReq] error:nil];
    for (VNRecognizedObjectObservation *ao in animReq.results) {
        VNClassificationObservation *lbl = ao.labels.firstObject;
        if (lbl != nil && lbl.confidence >= 0.35) {
            NSString *entry = [NSString stringWithFormat:@"%@ (%.0f%%)", [lbl.identifier lowercaseString], lbl.confidence * 100.0];
            [concreteObjects insertObject:entry atIndex:0];
        }
    }

    // 4. Face & human body detection pass.
    VNDetectFaceRectanglesRequest *faceReq = [[VNDetectFaceRectanglesRequest alloc] init];
    VNImageRequestHandler *faceHandler = [[VNImageRequestHandler alloc] initWithCGImage:cg
                                                                            orientation:kCGImagePropertyOrientationUp
                                                                                options:@{}];
    [faceHandler performRequests:@[faceReq] error:nil];
    NSUInteger faceCount = faceReq.results.count;

    VNDetectHumanRectanglesRequest *humanReq = [[VNDetectHumanRectanglesRequest alloc] init];
    VNImageRequestHandler *humanHandler = [[VNImageRequestHandler alloc] initWithCGImage:cg
                                                                             orientation:kCGImagePropertyOrientationUp
                                                                                 options:@{}];
    [humanHandler performRequests:@[humanReq] error:nil];
    NSUInteger humanCount = humanReq.results.count;

    NSString *tones = kronosSampleSurfaceToneSummary(cg);
    return @{
        @"objects": concreteObjects,
        @"ocrLines": ocrLines,
        @"ocrChars": @(totalOcrChars),
        @"faces": @(faceCount),
        @"humans": @(humanCount),
        @"tones": tones ?: @"natural lighting",
    };
}

static inline NSString *kronosSummarizePhotoSubjects(UIImage *img) {
    UIImage *upright = kronosRenderUprightImage(img, 1024.0);
    NSDictionary *rep = kronosAnalyzeUprightImageWithVision(upright);
    NSArray<NSString *> *objs = rep[@"objects"];
    NSUInteger faces = [rep[@"faces"] unsignedIntegerValue];
    NSUInteger humans = [rep[@"humans"] unsignedIntegerValue];
    NSMutableArray<NSString *> *cleanNames = [NSMutableArray array];
    for (NSString *o in objs) {
        NSRange paren = [o rangeOfString:@" ("];
        NSString *base = (paren.location != NSNotFound) ? [o substringToIndex:paren.location] : o;
        if (cleanNames.count < 3) [cleanNames addObject:base];
    }
    if (faces > 0 || humans > 0) {
        if (cleanNames.count > 0) {
            return [NSString stringWithFormat:@"portrait / person with %@", [cleanNames componentsJoinedByString:@", "]];
        }
        return (faces > 1) ? @"group photo" : @"portrait photo";
    }
    if (cleanNames.count > 0) {
        return [cleanNames componentsJoinedByString:@", "];
    }
    return rep[@"tones"] ?: @"today's workspace snapshot";
}

static inline NSString *kronosExtractVisionOcrAndLabels(UIImage *image) {
    UIImage *upright = kronosRenderUprightImage(image, 1280.0);
    NSDictionary *rep = kronosAnalyzeUprightImageWithVision(upright);
    NSArray<NSString *> *objects = rep[@"objects"];
    NSArray<NSString *> *ocrLines = rep[@"ocrLines"];
    NSUInteger ocrChars = [rep[@"ocrChars"] unsignedIntegerValue];
    NSUInteger faces = [rep[@"faces"] unsignedIntegerValue];
    NSUInteger humans = [rep[@"humans"] unsignedIntegerValue];
    NSString *tones = rep[@"tones"] ?: @"natural lighting";

    NSMutableString *info = [NSMutableString string];
    if (ocrChars < 35) {
        [info appendString:@"[Visual Perception — Physical Object / Scene Photograph]\n"];
        if (objects.count > 0) {
            [info appendFormat:@"- Detected objects and setting: %@\n", [objects componentsJoinedByString:@", "]];
        }
        if (faces > 0 || humans > 0) {
            [info appendFormat:@"- People in frame: %lu face(s) / person(s) visible\n", (unsigned long)MAX(faces, humans)];
        }
        [info appendFormat:@"- Surface colors & lighting: %@\n", tones];
        [info appendString:@"- Answering instruction: Identify and describe the physical object(s) and scene directly and naturally (for example, naming the laptop/MacBook, device, or object on the desk/surface). Do NOT describe this as a text detection or OCR task, and ignore tiny incidental keycap letters.\n"];
    } else {
        [info appendString:@"[Visual Perception — Document / Screen / Text-Rich Image]\n"];
        if (objects.count > 0) {
            [info appendFormat:@"- Visual layout / scene tags: %@\n", [objects componentsJoinedByString:@", "]];
        }
        [info appendFormat:@"- Colors & lighting: %@\n", tones];
        NSString *joined = [ocrLines componentsJoinedByString:@"\n"];
        if (joined.length > 6000) {
            joined = [[joined substringToIndex:6000] stringByAppendingString:@"\n...[truncated]"];
        }
        [info appendFormat:@"- Readable text in image:\n%@\n", joined];
    }
    return info;
}

static inline KronosAttachmentItem *kronosBuildAttachmentFromImage(UIImage *img, NSString *suggestedName, NSUInteger slot) {
    if (img == nil) return nil;
    UIImage *normalized = kronosRenderUprightImage(img, 1280.0);
    if (normalized == nil) return nil;
    NSData *pngData = UIImagePNGRepresentation(normalized);
    if (pngData == nil) return nil;

    NSString *name = suggestedName.length > 0 ? suggestedName : [NSString stringWithFormat:@"image_%lu.png", (unsigned long)slot];
    if (![[name lowercaseString] hasSuffix:@".png"] && ![[name lowercaseString] hasSuffix:@".jpg"] && ![[name lowercaseString] hasSuffix:@".jpeg"]) {
        name = [name stringByAppendingString:@".png"];
    }

    NSString *wsDir = kronosWorkspaceDirectory();
    NSString *savedPath = [wsDir stringByAppendingPathComponent:name];
    [pngData writeToFile:savedPath atomically:YES];

    NSString *visionInfo = kronosExtractVisionOcrAndLabels(normalized);

    KronosAttachmentItem *item = [[KronosAttachmentItem alloc] init];
    item.itemId = [[NSUUID UUID] UUIDString];
    item.filename = name;
    item.mimeType = @"image/png";
    item.rawData = pngData;
    item.thumbnail = normalized;
    item.isImage = YES;
    item.workspacePath = savedPath;
    item.summaryBadge = [NSString stringWithFormat:@"%.0fx%.0f · %@", normalized.size.width, normalized.size.height, kronosFormatBytes(pngData.length)];
    item.promptContext = [NSString stringWithFormat:@"[Attached Image: %@ (%.0fx%.0f, saved in workspace as %@)]\n%@",
                          name, normalized.size.width, normalized.size.height, name, visionInfo];
    return item;
}

static inline KronosAttachmentItem *kronosBuildAttachmentFromData(NSData *data, NSString *filename, NSString *mime, NSUInteger slot) {
    if (data == nil || data.length == 0) return nil;
    NSString *cleanName = filename.lastPathComponent.length > 0
        ? filename.lastPathComponent
        : [NSString stringWithFormat:@"file_%lu.txt", (unsigned long)slot];
    NSString *lower = [cleanName lowercaseString];

    if ([lower hasSuffix:@".png"] || [lower hasSuffix:@".jpg"] || [lower hasSuffix:@".jpeg"] ||
        [lower hasSuffix:@".heic"] || [lower hasSuffix:@".webp"] || [lower hasSuffix:@".gif"]) {
        UIImage *img = [UIImage imageWithData:data];
        if (img != nil) {
            return kronosBuildAttachmentFromImage(img, cleanName, slot);
        }
    }

    NSString *wsDir = kronosWorkspaceDirectory();
    NSString *savedPath = [wsDir stringByAppendingPathComponent:cleanName];
    [data writeToFile:savedPath atomically:YES];

    NSMutableString *promptCtx = [NSMutableString string];
    if ([lower hasSuffix:@".pdf"]) {
        PDFDocument *pdf = [[PDFDocument alloc] initWithData:data];
        NSUInteger pageCount = pdf ? pdf.pageCount : 0;
        NSMutableString *pdfText = [NSMutableString string];
        for (NSUInteger p = 0; p < MIN(pageCount, 25); p++) {
            PDFPage *page = [pdf pageAtIndex:p];
            NSString *pt = [page.string stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
            if (pt.length > 10) {
                [pdfText appendFormat:@"--- Page %lu ---\n%@\n", (unsigned long)(p + 1), pt];
            } else if (page != nil && p < 6) {
                CGRect box = [page boundsForBox:kPDFDisplayBoxMediaBox];
                CGSize sz = CGSizeMake(MAX(box.size.width, 600), MAX(box.size.height, 800));
                UIGraphicsImageRenderer *r = [[UIGraphicsImageRenderer alloc] initWithSize:sz];
                UIImage *pageImg = [r imageWithActions:^(UIGraphicsImageRendererContext *c) {
                    [[UIColor whiteColor] setFill];
                    UIRectFill(CGRectMake(0, 0, sz.width, sz.height));
                    CGContextTranslateCTM(c.CGContext, 0.0, sz.height);
                    CGContextScaleCTM(c.CGContext, 1.0, -1.0);
                    [page drawWithBox:kPDFDisplayBoxMediaBox toContext:c.CGContext];
                }];
                [pdfText appendFormat:@"--- Page %lu (OCR) ---\n%@\n", (unsigned long)(p + 1), kronosExtractVisionOcrAndLabels(pageImg)];
            }
            if (pdfText.length > 10000) break;
        }
        if (pdfText.length > 10000) {
            pdfText = [[pdfText substringToIndex:10000] mutableCopy];
            [pdfText appendString:@"\n...[truncated]"];
        }
        [promptCtx appendFormat:@"[Attached PDF: %@ (%lu pages, saved in workspace as %@)]\n%@\n",
         cleanName, (unsigned long)pageCount, cleanName, pdfText.length > 0 ? pdfText : @"(No text extracted)"];
    } else {
        char badgeBuf[256] = {0};
        char ctxBuf[16384] = {0};
        char relBuf[256] = {0};
        int32_t rc = kronos_ingest_attachment_bytes(
            [cleanName UTF8String],
            [(mime ?: @"") UTF8String],
            (const uint8_t *)data.bytes,
            data.length,
            [wsDir UTF8String],
            slot,
            badgeBuf, sizeof(badgeBuf),
            ctxBuf, sizeof(ctxBuf),
            relBuf, sizeof(relBuf)
        );
        if (rc == 0 && strlen(ctxBuf) > 0) {
            [promptCtx appendFormat:@"[Saved in workspace as %@]\n%s", cleanName, ctxBuf];
        } else {
            NSString *utf8 = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
            if (utf8.length > 10000) utf8 = [[utf8 substringToIndex:10000] stringByAppendingString:@"\n...[truncated]"];
            [promptCtx appendFormat:@"[Attached File: %@ (%@, saved in workspace as %@)]\n%@\n",
             cleanName, kronosFormatBytes(data.length), cleanName, utf8 ?: @"(binary file)"];
        }
    }

    KronosAttachmentItem *item = [[KronosAttachmentItem alloc] init];
    item.itemId = [[NSUUID UUID] UUIDString];
    item.filename = cleanName;
    item.mimeType = mime ?: @"application/octet-stream";
    item.rawData = data;
    item.isImage = NO;
    item.workspacePath = savedPath;
    item.summaryBadge = kronosFormatBytes(data.length);
    item.promptContext = promptCtx;
    return item;
}

// Intent Classification, Anti-Hallucination Evidence Gates, and User-Taught Local Fact Store.
static inline BOOL kronosIsInterrogativeOrLookupText(NSString *rawText) {
    NSString *norm = kronosNormalizeApostrophes(rawText ?: @"");
    NSString *low = [[norm lowercaseString] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (low.length == 0) return NO;
    if ([low hasSuffix:@"?"]) return YES;
    NSArray<NSString *> *qStarts = @[
        @"what ", @"what's ", @"whats ", @"who ", @"who's ", @"whos ",
        @"when ", @"where ", @"which ", @"why ", @"how ",
        @"did ", @"does ", @"do i ", @"do you know ", @"was ", @"were ",
        @"is ", @"are ", @"can you tell me what", @"tell me what", @"show me what", @"check what"
    ];
    for (NSString *qs in qStarts) {
        if ([low hasPrefix:qs]) return YES;
    }
    NSArray<NSString *> *lookupPhrases = @[
        @"last message", @"latest message", @"recent message", @"recent messages",
        @"unread message", @"sent me", @"messaged me", @"texted me",
        @"phone number", @"dads number", @"dad's number", @"moms number", @"mom's number"
    ];
    for (NSString *lp in lookupPhrases) {
        if ([low containsString:lp]) return YES;
    }
    return NO;
}

static inline BOOL kronosIsActionOrComposerCommand(NSString *rawText) {
    if (kronosIsInterrogativeOrLookupText(rawText)) return NO;
    NSString *lower = [[kronosNormalizeApostrophes(rawText ?: @"") lowercaseString]
                       stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if ([lower hasPrefix:@"open "] || [lower hasPrefix:@"launch "] || [lower hasPrefix:@"start "] ||
        [lower hasPrefix:@"write "] || [lower hasPrefix:@"create "] || [lower hasPrefix:@"make "] ||
        [lower hasPrefix:@"draft "] || [lower hasPrefix:@"send "] || [lower hasPrefix:@"message "] ||
        [lower hasPrefix:@"text "] || [lower containsString:@"open whatsapp"] ||
        [lower containsString:@"open whats app"] || [lower containsString:@"write a message"] ||
        [lower containsString:@"send a message"]) {
        return YES;
    }
    if (([lower containsString:@"linkedin"] || [lower containsString:@"linkendin"] ||
         [lower containsString:@"linked in"] || [lower containsString:@"linkedln"]) &&
        ([lower containsString:@"post"] || [lower containsString:@"picture"] || [lower containsString:@"photo"])) {
        return YES;
    }
    return NO;
}

static inline BOOL kronosIsExternalMessageLookupQuery(NSString *text) {
    NSString *lower = [[kronosNormalizeApostrophes(text ?: @"") lowercaseString]
                       stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (lower.length == 0 || kronosIsActionOrComposerCommand(lower)) return NO;

    // Never classify declarative statements ("This is my WhatsApp number ...", "My WhatsApp number is ...", "Remember ...")
    // or personal number/handle/account lookups ("What is my WhatsApp number?") as message-thread lookups.
    NSArray<NSString *> *declarativeStarts = @[
        @"this is ", @"here is ", @"my ", @"i am ", @"i'm ", @"i prefer ", @"i live ", @"i work ",
        @"remember ", @"please remember ", @"save ", @"note ", @"keep in mind", @"don't forget", @"dont forget",
        @"remind me to ", @"remind me that ", @"set a reminder", @"add a reminder", @"actually ", @"actually,",
        @"correction:", @"update:", @"fyi "
    ];
    for (NSString *ds in declarativeStarts) {
        if ([lower hasPrefix:ds]) return NO;
    }

    NSArray<NSString *> *nonMessageIdentifierPhrases = @[
        @"whatsapp number", @"whats app number", @"phone number", @"mobile number", @"contact number",
        @"my number", @"telegram number", @"telegram handle", @"whatsapp account", @"whatsapp link"
    ];
    for (NSString *p in nonMessageIdentifierPhrases) {
        if ([lower containsString:p]) return NO;
    }

    NSArray<NSString *> *cues = @[
        @"last message", @"latest message", @"recent message", @"recent messages",
        @"unread message", @"message from ", @"messages from ", @"message to ",
        @"messages to ", @"message with ", @"messages with ", @"whatsapp message",
        @"whatsapp messages", @"whatsapp chat", @"whats app message", @"whats app chat",
        @"imessage from ", @"imessages from ", @"chat with ", @"conversation with ",
        @"sent me", @"sent to ", @"send to ", @"sent on whatsapp", @"messaged me", @"texted me",
        @"replied to me", @"wrote to me", @"said to me", @"my reply to ",
        @"what did i send", @"what did i reply", @"what did i text", @"what did i message",
        @"on whatsapp", @"on whats app", @"on imessage", @"on telegram",
        @"said on whatsapp", @"say on whatsapp", @"wrote on whatsapp"
    ];
    for (NSString *c in cues) {
        if ([lower containsString:c]) return YES;
    }
    return NO;
}

static inline NSString *kronosUserFactsFilePath(void) {
    return [kronosDocumentsDirectory() stringByAppendingPathComponent:@"kronos_user_facts.json"];
}

static inline NSString *kronosDistilledVaultFilePath(void) {
    return [kronosDocumentsDirectory() stringByAppendingPathComponent:@"kronos_distilled_14b_vault.json"];
}

static inline NSMutableDictionary *kronosLoadMemoryStore(void) {
    NSData *d = [NSData dataWithContentsOfFile:kronosUserFactsFilePath()];
    NSMutableDictionary *store = [NSMutableDictionary dictionary];
    if (d.length > 0) {
        id raw = [NSJSONSerialization JSONObjectWithData:d options:NSJSONReadingMutableContainers error:nil];
        if ([raw isKindOfClass:[NSDictionary class]]) {
            NSDictionary *dict = (NSDictionary *)raw;
            if ([dict[@"facts"] isKindOfClass:[NSDictionary class]] ||
                [dict[@"notes"] isKindOfClass:[NSArray class]] ||
                [dict[@"reminders"] isKindOfClass:[NSArray class]]) {
                store = [dict mutableCopy];
            } else {
                // Migrate flat v1 key->string dictionary seamlessly into structured v2 memory store.
                NSMutableDictionary *migratedFacts = [NSMutableDictionary dictionary];
                for (NSString *k in dict.allKeys) {
                    id v = dict[k];
                    if ([v isKindOfClass:[NSString class]]) {
                        NSString *sv = (NSString *)v;
                        NSRange col = [sv rangeOfString:@": "];
                        NSString *lbl = (col.location != NSNotFound) ? [sv substringToIndex:col.location] : k;
                        NSString *val = (col.location != NSNotFound) ? [sv substringFromIndex:(col.location + 2)] : sv;
                        migratedFacts[k] = [@{@"label": lbl, @"value": val, @"ts": @(0)} mutableCopy];
                    }
                }
                store[@"facts"] = migratedFacts;
            }
        }
    }
    if (![store[@"facts"] isKindOfClass:[NSMutableDictionary class]]) {
        store[@"facts"] = [store[@"facts"] isKindOfClass:[NSDictionary class]] ? [store[@"facts"] mutableCopy] : [NSMutableDictionary dictionary];
    }
    if (![store[@"notes"] isKindOfClass:[NSMutableArray class]]) {
        store[@"notes"] = [store[@"notes"] isKindOfClass:[NSArray class]] ? [store[@"notes"] mutableCopy] : [NSMutableArray array];
    }
    if (![store[@"reminders"] isKindOfClass:[NSMutableArray class]]) {
        store[@"reminders"] = [store[@"reminders"] isKindOfClass:[NSArray class]] ? [store[@"reminders"] mutableCopy] : [NSMutableArray array];
    }
    return store;
}

static inline void kronosSaveMemoryStore(NSDictionary *store) {
    if (store == nil) return;
    NSData *out = [NSJSONSerialization dataWithJSONObject:store options:NSJSONWritingPrettyPrinted error:nil];
    if (out != nil) {
        [out writeToFile:kronosUserFactsFilePath() atomically:YES];
    }
}

static inline void kronosMergeMemoryStoreFromRemoteDict(NSDictionary *remote) {
    if (![remote isKindOfClass:[NSDictionary class]]) return;
    NSMutableDictionary *local = kronosLoadMemoryStore();
    NSMutableDictionary *localFacts = local[@"facts"];
    BOOL changed = NO;

    NSDictionary *remoteFacts = [remote[@"facts"] isKindOfClass:[NSDictionary class]] ? remote[@"facts"] : nil;
    for (NSString *k in remoteFacts.allKeys) {
        NSDictionary *rv = remoteFacts[k];
        if (![rv isKindOfClass:[NSDictionary class]]) continue;
        double rTs = [rv[@"ts"] respondsToSelector:@selector(doubleValue)] ? [rv[@"ts"] doubleValue] : 0.0;
        NSDictionary *lv = localFacts[k];
        double lTs = ([lv isKindOfClass:[NSDictionary class]] && [lv[@"ts"] respondsToSelector:@selector(doubleValue)]) ? [lv[@"ts"] doubleValue] : -1.0;
        if (lv == nil || rTs > lTs) {
            localFacts[k] = [rv mutableCopy];
            changed = YES;
        }
    }

    for (NSString *listKey in @[@"notes", @"reminders"]) {
        NSArray *remoteList = [remote[listKey] isKindOfClass:[NSArray class]] ? remote[listKey] : nil;
        if (remoteList.count == 0) continue;
        NSMutableArray *localList = local[listKey];
        for (NSDictionary *ri in remoteList) {
            if (![ri isKindOfClass:[NSDictionary class]]) continue;
            NSString *rt = ri[@"text"];
            if (![rt isKindOfClass:[NSString class]] || rt.length == 0) continue;
            BOOL exists = NO;
            for (NSDictionary *li in localList) {
                if ([[li[@"text"] ?: @"" lowercaseString] isEqualToString:[rt lowercaseString]]) {
                    exists = YES;
                    break;
                }
            }
            if (!exists) {
                [localList addObject:[ri mutableCopy]];
                changed = YES;
            }
        }
    }

    if (changed) {
        kronosSaveMemoryStore(local);
    }
}

static inline NSString *kronosCanonicalFactKey(NSString *rawSubject) {
    NSArray<NSString *> *words = kronosSplitAlphaWords(rawSubject ?: @"");
    NSSet<NSString *> *drop = [NSSet setWithArray:@[@"my", @"your", @"the", @"a", @"an", @"own", @"current", @"new", @"is", @"s"]];
    NSMutableArray<NSString *> *kept = [NSMutableArray array];
    for (NSString *w in words) {
        if (![drop containsObject:w]) {
            [kept addObject:w];
        }
    }
    return [kept componentsJoinedByString:@" "];
}

static inline NSString *kronosDisplayLabelForUserSubject(NSString *lhs) {
    NSString *clean = [lhs stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSString *low = [clean lowercaseString];
    if ([low hasPrefix:@"my "]) {
        return [NSString stringWithFormat:@"Your %@", [clean substringFromIndex:3]];
    }
    if ([low hasPrefix:@"your "]) {
        return clean;
    }
    return clean;
}

static inline BOOL kronosValueLooksTransientComplaint(NSString *rhs) {
    NSString *low = [[rhs ?: @"" lowercaseString] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSArray<NSString *> *transient = @[
        @"bugged", @"broken", @"not working", @"crashing", @"slow", @"wrong",
        @"hallucinating", @"missing", @"empty", @"offline", @"down"
    ];
    for (NSString *t in transient) {
        if ([low isEqualToString:t] || [low hasPrefix:[t stringByAppendingString:@" "]] || [low hasSuffix:[@" " stringByAppendingString:t]]) {
            return YES;
        }
    }
    return NO;
}

static inline BOOL kronosTrySaveUserTaughtFact(NSString *rawText, NSString **outConfirmation) {
    NSString *trimmed = [kronosNormalizeApostrophes(rawText ?: @"") stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (trimmed.length < 6 || trimmed.length > 260) return NO;
    if ([trimmed containsString:@"\n"] && trimmed.length > 160) return NO;
    NSString *low = [trimmed lowercaseString];
    if ([low hasSuffix:@"?"]) return NO;
    if (kronosIsActionOrComposerCommand(low)) return NO;

    // Never treat questions or recall prompts ("Remind me of my ...", "What is ...", "Tell me ...") as fact-write commands.
    NSArray<NSString *> *interrogativeStarts = @[
        @"what ", @"what's ", @"whats ", @"who ", @"where ", @"when ", @"why ", @"how ",
        @"which ", @"is ", @"are ", @"was ", @"were ", @"do ", @"does ", @"did ",
        @"can ", @"could ", @"would ", @"will ", @"should ", @"tell me", @"show me",
        @"give me", @"explain ", @"summarize ", @"help ", @"check ", @"find ", @"search ",
        @"remind me of ", @"remind me what ", @"remind me who ", @"remind me when ",
        @"remind me where ", @"remind me which ", @"remind me how ", @"remind me about my ",
        @"remind me about our ", @"remind me about the "
    ];
    for (NSString *iq in interrogativeStarts) {
        if ([low hasPrefix:iq]) return NO;
    }

    NSTimeInterval now = [[NSDate date] timeIntervalSince1970];

    // 1. Explicit action reminders ("Remind me to ...", "Don't forget to ...", "Set a reminder to ...").
    NSArray<NSString *> *reminderPrefixes = @[
        @"remind me to ", @"please remind me to ", @"don't forget to ", @"dont forget to ",
        @"set a reminder to ", @"set a reminder: ", @"add a reminder to ", @"add a reminder: "
    ];
    for (NSString *rp in reminderPrefixes) {
        if ([low hasPrefix:rp]) {
            NSString *task = [[trimmed substringFromIndex:rp.length]
                              stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@" \t\n\r."]];
            if (task.length < 3) return NO;
            NSMutableDictionary *store = kronosLoadMemoryStore();
            NSMutableArray *reminders = store[@"reminders"];
            BOOL exists = NO;
            for (NSDictionary *r in reminders) {
                if ([[r[@"text"] ?: @"" lowercaseString] isEqualToString:[task lowercaseString]]) {
                    exists = YES;
                    break;
                }
            }
            if (!exists) {
                [reminders addObject:@{@"text": task, @"ts": @(now)}];
                kronosSaveMemoryStore(store);
            }
            if (outConfirmation) {
                *outConfirmation = [NSString stringWithFormat:@"Saved to your local Kronos reminders: %@. Ask me about your reminders anytime and I will remind you.", task];
            }
            return YES;
        }
    }

    // 2. Explicit memory / note / correction prefixes.
    NSArray<NSString *> *explicitPrefixes = @[
        @"remember that ", @"please remember that ", @"please remember ", @"remember: ", @"remember ",
        @"keep in mind that ", @"keep in mind: ", @"keep in mind ",
        @"don't forget that ", @"dont forget that ", @"don't forget ", @"dont forget ",
        @"remind me that ", @"remind me: ",
        @"save this: ", @"save note: ", @"save fact: ", @"save that ",
        @"note that ", @"note: ", @"learn this: ", @"fyi: ", @"fyi ",
        @"correction: ", @"update: "
    ];
    NSString *explicitBody = nil;
    for (NSString *p in explicitPrefixes) {
        if ([low hasPrefix:p]) {
            explicitBody = [[trimmed substringFromIndex:p.length]
                            stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@" \t\n\r."]];
            break;
        }
    }
    if (explicitBody != nil) {
        if (explicitBody.length < 3) return NO;
        NSString *ebLow = [explicitBody lowercaseString];
        NSRange sep = [ebLow rangeOfString:@" is "];
        NSUInteger sepLen = 4;
        if (sep.location == NSNotFound) {
            sep = [ebLow rangeOfString:@": "];
            sepLen = 2;
        }
        if (sep.location == NSNotFound) {
            sep = [ebLow rangeOfString:@" = "];
            sepLen = 3;
        }
        NSMutableDictionary *store = kronosLoadMemoryStore();
        if (sep.location != NSNotFound && sep.location >= 2 && sep.location <= 56) {
            NSString *lhs = [[explicitBody substringToIndex:sep.location] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
            NSString *rhs = [[explicitBody substringFromIndex:(sep.location + sepLen)] stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@" \t\n\r."]];
            NSString *canon = kronosCanonicalFactKey(lhs);
            if (canon.length >= 2 && rhs.length > 0) {
                NSString *label = kronosDisplayLabelForUserSubject(lhs);
                store[@"facts"][canon] = @{@"label": label, @"value": rhs, @"ts": @(now)};
                kronosSaveMemoryStore(store);
                if (outConfirmation) {
                    *outConfirmation = [NSString stringWithFormat:@"Saved to your local Kronos memory: %@ is %@.", label, rhs];
                }
                return YES;
            }
        }
        NSMutableArray *notes = store[@"notes"];
        BOOL dup = NO;
        for (NSDictionary *n in notes) {
            if ([[n[@"text"] ?: @"" lowercaseString] isEqualToString:ebLow]) { dup = YES; break; }
        }
        if (!dup) {
            [notes addObject:@{@"text": explicitBody, @"ts": @(now)}];
            kronosSaveMemoryStore(store);
        }
        if (outConfirmation) {
            *outConfirmation = [NSString stringWithFormat:@"Saved to your local Kronos memory: %@.", explicitBody];
        }
        return YES;
    }

    // 3. Natural declarative "This is my <attribute> <value>" / "Here is my <attribute> <value>"
    //    (e.g. "This is my WhatsApp number 09033351286", "Here is my Dad's phone number +234803...").
    NSArray<NSString *> *thisIsMyPrefixes = @[@"this is my ", @"here is my ", @"this's my ", @"heres my ", @"here's my "];
    for (NSString *tp in thisIsMyPrefixes) {
        if ([low hasPrefix:tp]) {
            NSString *rest = [[trimmed substringFromIndex:tp.length]
                              stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@" \t\n\r."]];
            if (rest.length < 3) return NO;
            NSString *lhsAttr = nil;
            NSString *rhsVal = nil;
            NSString *restLow = [rest lowercaseString];
            for (NSString *delim in @[@": ", @" is ", @" = ", @" - "]) {
                NSRange r = [restLow rangeOfString:delim];
                if (r.location != NSNotFound && r.location >= 2 && r.location <= 48) {
                    lhsAttr = [[rest substringToIndex:r.location] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
                    rhsVal = [[rest substringFromIndex:(r.location + delim.length)] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
                    break;
                }
            }
            if (lhsAttr == nil) {
                NSArray<NSString *> *tokens = [rest componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
                NSMutableArray<NSString *> *nonEmpty = [NSMutableArray array];
                for (NSString *t in tokens) { if (t.length > 0) [nonEmpty addObject:t]; }
                if (nonEmpty.count >= 2 && nonEmpty.count <= 8) {
                    NSUInteger splitAt = NSNotFound;
                    for (NSUInteger i = 1; i < nonEmpty.count; i++) {
                        NSString *tok = nonEmpty[i];
                        BOOL hasDigitOrSym = ([tok rangeOfCharacterFromSet:[NSCharacterSet characterSetWithCharactersInString:@"0123456789@+/"]].location != NSNotFound);
                        if (hasDigitOrSym) {
                            splitAt = i;
                            break;
                        }
                    }
                    if (splitAt == NSNotFound && nonEmpty.count >= 3) {
                        NSString *prevLow = [nonEmpty[nonEmpty.count - 2] lowercaseString];
                        NSSet<NSString *> *attrNouns = [NSSet setWithArray:@[
                            @"number", @"phone", @"email", @"handle", @"username", @"code",
                            @"password", @"passcode", @"pin", @"id", @"address", @"name", @"key"
                        ]];
                        if ([attrNouns containsObject:prevLow]) {
                            splitAt = nonEmpty.count - 1;
                        }
                    }
                    if (splitAt != NSNotFound && splitAt >= 1 && splitAt <= 5) {
                        lhsAttr = [[nonEmpty subarrayWithRange:NSMakeRange(0, splitAt)] componentsJoinedByString:@" "];
                        rhsVal = [[nonEmpty subarrayWithRange:NSMakeRange(splitAt, nonEmpty.count - splitAt)] componentsJoinedByString:@" "];
                    }
                }
            }
            if (lhsAttr.length >= 2 && rhsVal.length >= 1) {
                NSString *fullLhs = [NSString stringWithFormat:@"My %@", lhsAttr];
                NSString *canon = kronosCanonicalFactKey(fullLhs);
                if (canon.length >= 2) {
                    NSString *label = kronosDisplayLabelForUserSubject(fullLhs);
                    NSMutableDictionary *store = kronosLoadMemoryStore();
                    store[@"facts"][canon] = @{@"label": label, @"value": rhsVal, @"ts": @(now)};
                    kronosSaveMemoryStore(store);
                    if (outConfirmation) {
                        *outConfirmation = [NSString stringWithFormat:@"Saved to your local Kronos memory: %@ is %@.", label, rhsVal];
                    }
                    return YES;
                }
            }
        }
    }

    // 4. Natural declarative "My <attribute> is <value>", "Actually my <attribute> is <value>", or "<Person>'s <attribute> is <value>".
    NSString *candidate = trimmed;
    NSString *candLow = low;
    if ([candLow hasPrefix:@"actually, "]) {
        candidate = [trimmed substringFromIndex:10];
        candLow = [candidate lowercaseString];
    } else if ([candLow hasPrefix:@"actually "]) {
        candidate = [trimmed substringFromIndex:9];
        candLow = [candidate lowercaseString];
    }

    BOOL startsPersonal = [candLow hasPrefix:@"my "] || [candLow hasPrefix:@"our "];
    NSRange isRange = [candLow rangeOfString:@" is "];
    BOOL startsPossessivePerson = (!startsPersonal && isRange.location != NSNotFound && isRange.location <= 36 &&
                                   [[candLow substringToIndex:isRange.location] containsString:@"'s "]);
    if ((startsPersonal || startsPossessivePerson) && isRange.location != NSNotFound && isRange.location >= 4 && isRange.location <= 46) {
        NSString *lhs = [[candidate substringToIndex:isRange.location] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
        NSString *rhs = [[candidate substringFromIndex:(isRange.location + 4)]
                         stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@" \t\n\r."]];
        NSArray<NSString *> *lhsWords = kronosSplitAlphaWords(lhs);
        if (lhsWords.count >= 2 && lhsWords.count <= 6 && rhs.length >= 1 && rhs.length <= 140 && !kronosValueLooksTransientComplaint(rhs)) {
            NSString *canon = kronosCanonicalFactKey(lhs);
            if (canon.length >= 2) {
                NSString *label = kronosDisplayLabelForUserSubject(lhs);
                NSMutableDictionary *store = kronosLoadMemoryStore();
                store[@"facts"][canon] = @{@"label": label, @"value": rhs, @"ts": @(now)};
                kronosSaveMemoryStore(store);
                if (outConfirmation) {
                    *outConfirmation = [NSString stringWithFormat:@"Saved to your local Kronos memory: %@ is %@.", label, rhs];
                }
                return YES;
            }
        }
    }

    // 5. User preference statements ("I prefer <preference>").
    if ([candLow hasPrefix:@"i prefer "] && candidate.length <= 140) {
        NSString *pref = [[candidate substringFromIndex:9] stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@" \t\n\r."]];
        if (pref.length >= 3) {
            NSMutableDictionary *store = kronosLoadMemoryStore();
            NSMutableArray *notes = store[@"notes"];
            NSString *entry = [NSString stringWithFormat:@"User prefers %@", pref];
            BOOL dup = NO;
            for (NSDictionary *n in notes) {
                if ([[n[@"text"] ?: @"" lowercaseString] isEqualToString:[entry lowercaseString]]) { dup = YES; break; }
            }
            if (!dup) {
                [notes addObject:@{@"text": entry, @"ts": @(now)}];
                kronosSaveMemoryStore(store);
            }
            if (outConfirmation) {
                *outConfirmation = [NSString stringWithFormat:@"Saved your preference locally: you prefer %@.", pref];
            }
            return YES;
        }
    }

    return NO;
}

static inline void kronosBackfillLearnedMemoryFromConversations(NSArray<NSDictionary *> *conversations) {
    if (conversations.count == 0) return;
    // Replay user turns oldest-to-newest so newer corrections supersede older facts.
    for (NSDictionary *conv in [conversations reverseObjectEnumerator]) {
        NSArray *msgs = conv[@"messages"];
        if (![msgs isKindOfClass:[NSArray class]]) continue;
        for (NSDictionary *m in msgs) {
            if (![m isKindOfClass:[NSDictionary class]]) continue;
            if (![m[@"role"] isEqualToString:@"user"]) continue;
            NSString *txt = m[@"displayText"] ?: m[@"content"];
            if ([txt isKindOfClass:[NSString class]] && txt.length > 0) {
                kronosTrySaveUserTaughtFact(txt, NULL);
            }
        }
    }
}

static inline NSString *kronosLookupVerifiedLocalFact(NSString *rawQueryOrTopic) {
    NSString *norm = [kronosNormalizeApostrophes(rawQueryOrTopic ?: @"") stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSString *low = [norm lowercaseString];
    if (low.length < 2 || kronosIsActionOrComposerCommand(low)) return nil;

    NSMutableDictionary *store = kronosLoadMemoryStore();
    NSDictionary *facts = store[@"facts"];
    NSArray *notes = store[@"notes"];
    NSArray *reminders = store[@"reminders"];

    // 1. Direct reminder list query ("What are my reminders?", "What did I ask you to remind me to do?").
    if ([low containsString:@"my reminder"] || [low containsString:@"what reminder"] ||
        [low containsString:@"remind me to do"] || [low containsString:@"ask you to remind"]) {
        if (reminders.count == 0) {
            return @"You do not have any saved Kronos reminders yet. Tell me \"Remind me to ...\" anytime and I will save it locally.";
        }
        NSMutableString *out = [NSMutableString stringWithString:@"Here are your saved local Kronos reminders:\n"];
        for (NSDictionary *r in reminders) {
            NSString *rt = r[@"text"];
            if (rt.length > 0) [out appendFormat:@"\n- %@", rt];
        }
        return out;
    }

    // 2. Canonical fact key lookup ("What is my WhatsApp number?", "Remind me of my WhatsApp number", "When is Dera's birthday?").
    NSSet<NSString *> *queryStop = [NSSet setWithArray:@[
        @"what", @"whats", @"when", @"where", @"who", @"which", @"is", @"was", @"are", @"were",
        @"my", @"your", @"the", @"a", @"an", @"tell", @"me", @"show", @"give", @"please",
        @"do", @"does", @"did", @"you", @"i", @"remember", @"know", @"remind", @"of", @"about",
        @"that", @"for", @"s", @"say", @"said", @"saved", @"stored"
    ]];
    NSArray<NSString *> *rawWords = kronosSplitAlphaWords(low);
    NSMutableArray<NSString *> *qTokens = [NSMutableArray array];
    for (NSString *w in rawWords) {
        if (![queryStop containsObject:w] && ![qTokens containsObject:w]) {
            [qTokens addObject:w];
        }
    }
    if (qTokens.count == 0) return nil;

    NSString *joinedQueryKey = [qTokens componentsJoinedByString:@" "];
    NSDictionary *exactEntry = facts[joinedQueryKey];
    if ([exactEntry isKindOfClass:[NSDictionary class]]) {
        return [NSString stringWithFormat:@"%@ is %@.", exactEntry[@"label"] ?: joinedQueryKey, exactEntry[@"value"] ?: @""];
    }

    // Score structured facts by bidirectional token coverage so "What is my WhatsApp?" or "What is my WhatsApp phone number?" matches "whatsapp number".
    NSString *bestFactKey = nil;
    double bestScore = 0.0;
    for (NSString *k in facts.allKeys) {
        NSArray<NSString *> *kTokens = kronosSplitAlphaWords(k);
        if (kTokens.count == 0) continue;
        NSUInteger hits = 0;
        for (NSString *qt in qTokens) {
            if ([kTokens containsObject:qt]) hits++;
        }
        if (hits == 0) continue;
        double qCov = (double)hits / (double)qTokens.count;
        double kCov = (double)hits / (double)kTokens.count;
        double score = 0.55 * qCov + 0.45 * kCov;
        if (hits == kTokens.count || (hits == qTokens.count && hits >= 1 && kTokens.count <= 3)) {
            if (score > bestScore && score >= 0.65) {
                bestScore = score;
                bestFactKey = k;
            }
        }
    }
    if (bestFactKey != nil) {
        NSDictionary *entry = facts[bestFactKey];
        if ([entry isKindOfClass:[NSDictionary class]]) {
            return [NSString stringWithFormat:@"%@ is %@.", entry[@"label"] ?: bestFactKey, entry[@"value"] ?: @""];
        }
    }

    // 3. Search saved free-form notes & reminders by token overlap.
    for (NSDictionary *n in [notes reverseObjectEnumerator]) {
        NSString *nt = n[@"text"] ?: @"";
        NSArray<NSString *> *nWords = kronosSplitAlphaWords(nt);
        NSUInteger hits = 0;
        for (NSString *qt in qTokens) {
            if ([nWords containsObject:qt]) hits++;
        }
        if (hits > 0 && ((qTokens.count == 1 && hits == 1 && [qTokens.firstObject length] >= 4) || (hits * 2 >= qTokens.count && hits >= 2))) {
            return [NSString stringWithFormat:@"From your saved local Kronos memory:\n\n%@", nt];
        }
    }
    for (NSDictionary *r in [reminders reverseObjectEnumerator]) {
        NSString *rt = r[@"text"] ?: @"";
        NSArray<NSString *> *rWords = kronosSplitAlphaWords(rt);
        NSUInteger hits = 0;
        for (NSString *qt in qTokens) {
            if ([rWords containsObject:qt]) hits++;
        }
        if (hits >= 2 || (qTokens.count == 1 && hits == 1 && [qTokens.firstObject length] >= 4)) {
            return [NSString stringWithFormat:@"From your saved local Kronos reminders:\n\n%@", rt];
        }
    }

    // 4. Also check text/markdown/json files in KronosWorkspace on device when topic has >= 2 tokens or >= 4 chars.
    if (joinedQueryKey.length >= 4) {
        NSString *ws = kronosWorkspaceDirectory();
        NSArray<NSString *> *files = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:ws error:nil] ?: @[];
        for (NSString *fn in files) {
            NSString *lfn = [fn lowercaseString];
            if (!([lfn hasSuffix:@".txt"] || [lfn hasSuffix:@".md"] || [lfn hasSuffix:@".json"] || [lfn hasSuffix:@".csv"])) continue;
            NSString *content = [NSString stringWithContentsOfFile:[ws stringByAppendingPathComponent:fn] encoding:NSUTF8StringEncoding error:nil];
            if (content.length == 0) continue;
            for (NSString *line in [content componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]]) {
                if ([[line lowercaseString] containsString:joinedQueryKey] && line.length >= joinedQueryKey.length + 2) {
                    return [NSString stringWithFormat:@"Found in your workspace file %@:\n\n%@", fn, [line stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]]];
                }
            }
        }
    }
    return nil;
}

static inline NSString *kronosBuildLearnedMemoryPromptBlock(NSString *userQuery) {
    NSMutableDictionary *store = kronosLoadMemoryStore();
    NSDictionary *facts = store[@"facts"];
    NSArray *notes = store[@"notes"];
    NSArray *reminders = store[@"reminders"];
    if (facts.count == 0 && notes.count == 0 && reminders.count == 0) return @"";

    NSArray<NSString *> *qWords = kronosSplitAlphaWords(userQuery ?: @"");
    NSMutableArray<NSString *> *lines = [NSMutableArray array];
    for (NSString *k in facts.allKeys) {
        NSDictionary *e = facts[k];
        if (![e isKindOfClass:[NSDictionary class]]) continue;
        NSString *line = [NSString stringWithFormat:@"- %@: %@", e[@"label"] ?: k, e[@"value"] ?: @""];
        NSArray<NSString *> *kWords = kronosSplitAlphaWords(k);
        BOOL hit = (facts.count <= 12);
        for (NSString *qw in qWords) {
            if (qw.length >= 3 && [kWords containsObject:qw]) { hit = YES; break; }
        }
        if (hit) [lines addObject:line];
        if (lines.count >= 12) break;
    }
    NSUInteger noteCount = 0;
    for (NSDictionary *n in [notes reverseObjectEnumerator]) {
        NSString *nt = n[@"text"] ?: @"";
        if (nt.length > 0) {
            [lines addObject:[NSString stringWithFormat:@"- Note: %@", nt]];
            if (++noteCount >= 5) break;
        }
    }
    NSUInteger remCount = 0;
    for (NSDictionary *r in [reminders reverseObjectEnumerator]) {
        NSString *rt = r[@"text"] ?: @"";
        if (rt.length > 0) {
            [lines addObject:[NSString stringWithFormat:@"- Reminder: %@", rt]];
            if (++remCount >= 4) break;
        }
    }
    if (lines.count == 0) return @"";
    return [NSString stringWithFormat:@"[Verified Learned User Memory]\n%@\n", [lines componentsJoinedByString:@"\n"]];
}

// Online 14B Mac Teacher -> 1.5B iPhone Student Trajectory Distillation Vault.
// Captures verified 14B Teacher completions on Wi-Fi and retrieves top-matching Teacher demonstrations
// by token Jaccard/coverage similarity to calibrate offline 1.5B-Instruct generation.
static inline NSMutableArray<NSDictionary *> *kronosLoadDistilledTeacherVault(void) {
    NSData *d = [NSData dataWithContentsOfFile:kronosDistilledVaultFilePath()];
    if (d.length > 0) {
        id raw = [NSJSONSerialization JSONObjectWithData:d options:NSJSONReadingMutableContainers error:nil];
        if ([raw isKindOfClass:[NSArray class]]) {
            return [(NSArray *)raw mutableCopy];
        }
    }
    return [@[
        @{
            @"q": @"Explain how to structure a clean REST endpoint error response in JSON.",
            @"a": @"Return an appropriate HTTP 4xx or 5xx status code alongside a consistent JSON envelope containing a machine-readable `code`, a concise `message`, and an optional `request_id` for tracing. Avoid leaking internal stack traces or database query details in client-facing responses.",
            @"tokens": @[@"explain", @"structure", @"clean", @"rest", @"endpoint", @"error", @"response", @"json"]
        },
        @{
            @"q": @"Summarize key priorities when optimizing an iOS app for smooth scrolling and low memory.",
            @"a": @"Keep image decoding and Vision or network work off the main thread, downsample large photos to display resolution before rendering, reuse table and collection cells without allocating views during scroll, and bound in-memory caches so dirty RAM stays comfortably below the iOS Jetsam ceiling.",
            @"tokens": @[@"summarize", @"priorities", @"optimizing", @"ios", @"app", @"smooth", @"scrolling", @"memory"]
        }
    ] mutableCopy];
}

static inline void kronosRecordTeacherDistillationPair(NSString *userQuery, NSString *verified14BReply) {
    NSString *q = [userQuery ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSString *a = [verified14BReply ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (q.length < 10 || q.length > 360 || a.length < 24 || a.length > 1400) return;
    if ([q hasPrefix:@"[Attached "] || [q hasPrefix:@"[Device Context"]) return;
    NSArray<NSString *> *qTokens = kronosSplitAlphaWords(q);
    if (qTokens.count < 3) return;

    NSMutableArray<NSDictionary *> *vault = kronosLoadDistilledTeacherVault();
    NSSet<NSString *> *qSet = [NSSet setWithArray:qTokens];
    NSInteger replaceIdx = -1;
    for (NSUInteger i = 0; i < vault.count; i++) {
        NSArray<NSString *> *et = vault[i][@"tokens"] ?: @[];
        if (et.count == 0) continue;
        NSUInteger inter = 0;
        for (NSString *t in et) { if ([qSet containsObject:t]) inter++; }
        NSUInteger uni = qSet.count + et.count - inter;
        double jacc = uni > 0 ? ((double)inter / (double)uni) : 0.0;
        if (jacc >= 0.76) {
            replaceIdx = (NSInteger)i;
            break;
        }
    }
    NSDictionary *entry = @{
        @"q": q,
        @"a": a,
        @"tokens": qTokens,
        @"ts": @([[NSDate date] timeIntervalSince1970])
    };
    if (replaceIdx >= 0) {
        vault[(NSUInteger)replaceIdx] = entry;
    } else {
        [vault addObject:entry];
        if (vault.count > 160) {
            [vault removeObjectAtIndex:2];
        }
    }
    NSData *out = [NSJSONSerialization dataWithJSONObject:vault options:0 error:nil];
    if (out != nil) {
        [out writeToFile:kronosDistilledVaultFilePath() atomically:YES];
    }
}

static inline void kronosMergeDistilledVaultFromRemoteArray(NSArray *remote) {
    if (![remote isKindOfClass:[NSArray class]] || remote.count == 0) return;
    NSMutableArray<NSDictionary *> *vault = kronosLoadDistilledTeacherVault();
    BOOL changed = NO;
    for (NSDictionary *ri in remote) {
        if (![ri isKindOfClass:[NSDictionary class]]) continue;
        NSString *rq = ri[@"q"];
        NSString *ra = ri[@"a"];
        if (![rq isKindOfClass:[NSString class]] || ![ra isKindOfClass:[NSString class]] || rq.length < 8 || ra.length < 16) continue;
        NSArray<NSString *> *rtoks = [ri[@"tokens"] isKindOfClass:[NSArray class]] ? ri[@"tokens"] : kronosSplitAlphaWords(rq);
        NSDictionary *normalized = @{
            @"q": rq,
            @"a": ra,
            @"tokens": rtoks,
            @"ts": ri[@"ts"] ?: @([[NSDate date] timeIntervalSince1970])
        };
        NSInteger foundIdx = -1;
        for (NSUInteger i = 0; i < vault.count; i++) {
            if ([[vault[i][@"q"] ?: @"" lowercaseString] isEqualToString:[rq lowercaseString]]) {
                foundIdx = (NSInteger)i;
                break;
            }
        }
        if (foundIdx >= 0) {
            vault[(NSUInteger)foundIdx] = normalized;
            changed = YES;
        } else {
            [vault addObject:normalized];
            if (vault.count > 200) {
                [vault removeObjectAtIndex:2];
            }
            changed = YES;
        }
    }
    if (changed) {
        NSData *out = [NSJSONSerialization dataWithJSONObject:vault options:0 error:nil];
        if (out != nil) {
            [out writeToFile:kronosDistilledVaultFilePath() atomically:YES];
        }
    }
}

static inline NSArray<NSDictionary *> *kronosRetrieveDistilledTeacherShots(NSString *userQuery, NSUInteger maxShots) {
    NSArray<NSString *> *qTokens = kronosSplitAlphaWords(userQuery ?: @"");
    if (qTokens.count < 2 || maxShots == 0) return @[];
    NSSet<NSString *> *stops = [NSSet setWithArray:@[@"what", @"how", @"why", @"can", @"you", @"the", @"for", @"with", @"and", @"that", @"this", @"from", @"are", @"is"]];
    NSMutableSet<NSString *> *qSet = [NSMutableSet set];
    for (NSString *t in qTokens) {
        if (t.length >= 3 && ![stops containsObject:t]) [qSet addObject:t];
    }
    if (qSet.count == 0) return @[];

    NSArray<NSDictionary *> *vault = kronosLoadDistilledTeacherVault();
    NSMutableArray<NSDictionary *> *scored = [NSMutableArray array];
    for (NSDictionary *e in vault) {
        NSArray<NSString *> *et = e[@"tokens"] ?: @[];
        NSUInteger hits = 0;
        for (NSString *t in et) {
            if ([qSet containsObject:t]) hits++;
        }
        if (hits == 0) continue;
        double score = (double)hits / (double)MAX(qSet.count, (NSUInteger)3);
        if (score >= 0.28 && score < 0.98) {
            [scored addObject:@{@"score": @(score), @"entry": e}];
        }
    }
    [scored sortUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
        return [b[@"score"] compare:a[@"score"]];
    }];
    NSMutableArray<NSDictionary *> *shots = [NSMutableArray array];
    NSUInteger limit = MIN(maxShots, scored.count);
    for (NSUInteger i = 0; i < limit; i++) {
        NSDictionary *e = scored[i][@"entry"];
        [shots addObject:@{@"role": @"user", @"content": e[@"q"] ?: @""}];
        [shots addObject:@{@"role": @"assistant", @"content": e[@"a"] ?: @""}];
    }
    return shots;
}

static inline BOOL kronosIsContactPhoneOrEmailQuery(NSString *rawText, NSString **outTargetPerson) {
    NSString *norm = [kronosNormalizeApostrophes(rawText ?: @"") stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSString *low = [norm lowercaseString];
    if (low.length == 0 || kronosIsActionOrComposerCommand(low) || kronosIsExternalMessageLookupQuery(low)) return NO;

    BOOL asksPhoneOrContact = [low containsString:@"phone number"] || [low containsString:@"phone no"] ||
                              [low containsString:@"telephone"] || [low containsString:@"mobile number"] ||
                              [low containsString:@"contact number"] || [low containsString:@"digits"] ||
                              [low containsString:@"'s number"] || [low containsString:@"s number"] ||
                              [low containsString:@"dad phone"] || [low containsString:@"dads phone"] ||
                              [low containsString:@"dad number"] || [low containsString:@"dads number"] ||
                              [low containsString:@"mom phone"] || [low containsString:@"moms phone"] ||
                              [low containsString:@"mum phone"] || [low containsString:@"mums phone"] ||
                              [low containsString:@"'s email"] || [low containsString:@"email for "];
    if (!asksPhoneOrContact) return NO;

    // Exclude Melody's own verified public email ("what is my email") and non-contact/own-account identifiers.
    if ([low isEqualToString:@"what is my email"] || [low isEqualToString:@"what's my email"] || [low isEqualToString:@"whats my email"]) {
        return NO;
    }
    NSSet<NSString *> *nonPersonNouns = [NSSet setWithArray:@[
        @"account", @"routing", @"iban", @"bvn", @"nin", @"myn", @"ssn", @"tin", @"pin",
        @"passport", @"license", @"card", @"serial", @"imei", @"order", @"tracking", @"flight", @"port", @"version",
        @"whatsapp", @"telegram", @"signal", @"imessage", @"slack", @"discord"
    ]];
    NSArray<NSString *> *words = kronosSplitAlphaWords(low);
    for (NSString *w in words) {
        if ([nonPersonNouns containsObject:w]) return NO;
    }

    // First-person own-number queries ("what is my phone number", "what is my work phone number") belong to user memory, not third-party Contacts, unless referencing a relation ("my dad's phone number").
    NSSet<NSString *> *relationWords = [NSSet setWithArray:@[
        @"dad", @"dads", @"daddy", @"father", @"mom", @"moms", @"mum", @"mums", @"mother",
        @"sister", @"sisters", @"brother", @"brothers", @"wife", @"husband", @"partner",
        @"friend", @"boss", @"doctor", @"lawyer", @"son", @"daughter", @"uncle", @"aunt"
    ]];
    if ([words containsObject:@"my"] && ![low containsString:@"'s"]) {
        BOOL hasRelation = NO;
        for (NSString *w in words) {
            if ([relationWords containsObject:w]) { hasRelation = YES; break; }
        }
        if (!hasRelation) return NO;
    }

    NSSet<NSString *> *stripWords = [NSSet setWithArray:@[
        @"what", @"whats", @"is", @"was", @"are", @"the", @"my", @"a", @"an",
        @"phone", @"telephone", @"mobile", @"cell", @"contact", @"number", @"no",
        @"digits", @"email", @"address", @"info", @"information", @"for", @"of",
        @"please", @"tell", @"me", @"give", @"show", @"check", @"do", @"you", @"have", @"know", @"s"
    ]];
    NSMutableArray<NSString *> *personTokens = [NSMutableArray array];
    for (NSString *origTok in [norm componentsSeparatedByCharactersInSet:[[NSCharacterSet alphanumericCharacterSet] invertedSet]]) {
        if (origTok.length == 0) continue;
        NSString *lt = [origTok lowercaseString];
        if ([stripWords containsObject:lt]) continue;
        if ([lt isEqualToString:@"dads"]) {
            [personTokens addObject:@"Dad"];
        } else if ([lt isEqualToString:@"moms"] || [lt isEqualToString:@"mums"]) {
            [personTokens addObject:@"Mom"];
        } else {
            [personTokens addObject:origTok];
        }
    }
    if (personTokens.count == 0) return NO;
    NSString *target = [personTokens componentsJoinedByString:@" "];
    if ([[target lowercaseString] isEqualToString:target]) {
        target = [target capitalizedString];
    }
    if (outTargetPerson) *outTargetPerson = target;
    return YES;
}

static inline BOOL kronosQueryAsksConversationMemory(NSString *text) {
    NSString *lower = [[kronosNormalizeApostrophes(text ?: @"") lowercaseString]
                       stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (lower.length == 0 || kronosIsActionOrComposerCommand(lower) || kronosIsExternalMessageLookupQuery(lower)) {
        return NO;
    }
    NSArray<NSString *> *cues = @[
        @"remind me", @"remind myself", @"important conversation", @"important conversations",
        @"previous conversation", @"previous conversations", @"past conversation", @"earlier conversation",
        @"what did we talk", @"what did we discuss", @"what we talked", @"what we discussed",
        @"get context", @"my conversations", @"conversation history", @"our last chat"
    ];
    for (NSString *c in cues) {
        if ([lower containsString:c]) return YES;
    }
    return NO;
}

static inline BOOL kronosQueryAsksCreatorOrProfile(NSString *text) {
    NSString *lower = [kronosNormalizeApostrophes(text ?: @"") lowercaseString];
    if (kronosIsActionOrComposerCommand(lower)) return NO;
    NSArray<NSString *> *keys = @[
        @"who built", @"who created", @"who made", @"who wrote", @"who designed", @"who invented",
        @"your creator", @"your maker", @"your founder", @"your author", @"your developer", @"your builder",
        @"created you", @"made you", @"built you", @"creator of kronos", @"maker of kronos", @"who are you",
        @"melody", @"leonard", @"oparaugo", @"ipmscoutek",
        @"about me", @"who am i", @"know about me", @"my profile",
        @"github", @"linkedin", @"my email", @"twitter", @"x handle",
        @"my experience", @"years of experience", @"how long have i",
        @"my stack", @"tech stack", @"my repos", @"my profession", @"my name"
    ];
    for (NSString *k in keys) {
        if ([lower containsString:k]) {
            return YES;
        }
    }
    return NO;
}

static inline BOOL kronosIsUnverifiedPersonalFactQuery(NSString *rawText, NSString **outFactTopic) {
    NSString *norm = [kronosNormalizeApostrophes(rawText ?: @"") stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSString *low = [norm lowercaseString];
    if (low.length == 0 || kronosIsActionOrComposerCommand(low) ||
        kronosIsExternalMessageLookupQuery(low) || kronosQueryAsksConversationMemory(low) ||
        kronosQueryAsksCreatorOrProfile(low) || kronosIsContactPhoneOrEmailQuery(norm, NULL)) {
        return NO;
    }
    if ([low containsString:@"my photos"] || [low containsString:@"my contacts"] ||
        [low containsString:@"my workspace"] || [low containsString:@"my iphone"] ||
        [low containsString:@"my device"]) {
        return NO;
    }

    NSArray<NSString *> *myPrefixes = @[
        @"what is my ", @"what's my ", @"whats my ",
        @"tell me my ", @"do you know my ", @"show me my ", @"what was my "
    ];
    for (NSString *mp in myPrefixes) {
        if ([low hasPrefix:mp]) {
            NSString *topic = [[norm substringFromIndex:mp.length]
                               stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@" \t\n\r?."]];
            if (topic.length > 0) {
                if (outFactTopic) *outFactTopic = topic;
                return YES;
            }
        }
    }
    return NO;
}

static inline NSString *kronosStripDigitFormatting(NSString *text) {
    if (text.length == 0) return @"";
    NSMutableString *out = [NSMutableString stringWithCapacity:text.length];
    for (NSUInteger i = 0; i < text.length; i++) {
        unichar ch = [text characterAtIndex:i];
        if (ch == '-' || ch == ' ' || ch == '(' || ch == ')' || ch == '.' || ch == '+') {
            continue;
        }
        [out appendFormat:@"%C", ch];
    }
    return out;
}

static inline BOOL kronosContainsUngroundedPlaceholderDigits(NSString *reply, NSString *groundingContext, NSString *userQuery) {
    if (reply.length == 0) return NO;
    NSString *qLow = [userQuery ?: @"" lowercaseString];
    if ([qLow containsString:@"example"] || [qLow containsString:@"code"] || [qLow containsString:@"function"] ||
        [qLow containsString:@"html"] || [qLow containsString:@"script"] || [qLow containsString:@"math"] ||
        [qLow containsString:@"calculate"]) {
        return NO;
    }
    NSString *rLow = [reply lowercaseString];
    NSArray<NSString *> *classicFakes = @[
        @"12345", @"123456", @"1234567", @"12345678", @"123456789", @"1234567890",
        @"555-1234", @"555-0199", @"987654321", @"000000"
    ];
    for (NSString *fake in classicFakes) {
        if ([rLow containsString:fake] && ![(groundingContext ?: @"") containsString:fake] && ![qLow containsString:fake]) {
            return YES;
        }
    }

    // On personal phone/contact/identifier queries, reject any 6+ digit sequence absent from verified grounding context.
    NSArray<NSString *> *qWords = kronosSplitAlphaWords(qLow);
    NSSet<NSString *> *sensitiveNouns = [NSSet setWithArray:@[
        @"phone", @"telephone", @"mobile", @"cell", @"number", @"contact",
        @"dad", @"dads", @"daddy", @"father", @"mom", @"moms", @"mum", @"mums", @"mother",
        @"bvn", @"nin", @"myn", @"ssn", @"tin", @"pin", @"iban", @"routing", @"account", @"passport"
    ]];
    BOOL asksSensitiveNumber = NO;
    for (NSString *w in qWords) {
        if ([sensitiveNouns containsObject:w]) {
            asksSensitiveNumber = YES;
            break;
        }
    }
    if (asksSensitiveNumber) {
        NSString *groundNorm = kronosStripDigitFormatting(groundingContext ?: @"");
        NSString *queryNorm = kronosStripDigitFormatting(qLow);
        NSMutableString *run = [NSMutableString string];
        for (NSUInteger i = 0; i <= reply.length; i++) {
            unichar ch = (i < reply.length) ? [reply characterAtIndex:i] : 0;
            BOOL isDigit = (ch >= '0' && ch <= '9');
            BOOL isSep = (ch == '-' || ch == ' ' || ch == '(' || ch == ')' || ch == '.');
            if (isDigit) {
                [run appendFormat:@"%C", ch];
            } else if (isSep && run.length > 0 && (i + 1 < reply.length)) {
                unichar nextCh = [reply characterAtIndex:(i + 1)];
                if (!(nextCh >= '0' && nextCh <= '9') && nextCh != '(' && nextCh != ' ') {
                    if (run.length >= 6 && ![groundNorm containsString:run] && ![queryNorm containsString:run]) {
                        return YES;
                    }
                    [run setString:@""];
                }
            } else {
                if (run.length >= 6 && ![groundNorm containsString:run] && ![queryNorm containsString:run]) {
                    return YES;
                }
                [run setString:@""];
            }
        }
    }
    return NO;
}

// Detects degenerate or role-swapped prior turns in saved session history so they are excluded
// from multi-turn few-shot context and can never poison subsequent generations in an existing thread.
static inline BOOL kronosIsPoisonedHistoryReply(NSString *reply) {
    NSString *trimmed = [[kronosNormalizeApostrophes(reply ?: @"") stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] lowercaseString];
    if (trimmed.length == 0) return YES;
    if (kronosContainsUngroundedPlaceholderDigits(trimmed, @"", @"")) return YES;
    NSArray<NSString *> *badRoleSwaps = @[
        @"i created kronos", @"i made kronos", @"i built kronos",
        @"i am melody", @"i'm melody",
        @"my dads number is", @"my dad's number is", @"my dads phone number is", @"my dad's phone number is",
        @"my moms number is", @"my mom's number is", @"my myn is", @"my phone number is"
    ];
    for (NSString *b in badRoleSwaps) {
        if ([trimmed containsString:b]) return YES;
    }
    if ([trimmed isEqualToString:@"i am kronos."] || [trimmed isEqualToString:@"i am kronos"] ||
        [trimmed isEqualToString:@"melody leonard"] || [trimmed isEqualToString:@"melody leonard."] ||
        [trimmed isEqualToString:@"stopped."] || [trimmed isEqualToString:@"ready whenever you are."]) {
        return YES;
    }
    return NO;
}

static inline NSString *kronosVerifyAndRepairIdentityReply(NSString *reply, NSString *userQuery) {
    NSString *clean = [kronosSanitizeFirstPersonReply(reply ?: @"") stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (!kronosQueryAsksCreatorOrProfile(userQuery)) return clean;

    NSString *qLow = [[kronosNormalizeApostrophes(userQuery ?: @"") lowercaseString]
                      stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    BOOL asksCreator = [qLow containsString:@"who created"] || [qLow containsString:@"who made"] ||
                       [qLow containsString:@"who built"] || [qLow containsString:@"who designed"] ||
                       [qLow containsString:@"who wrote"] || [qLow containsString:@"your maker"] ||
                       [qLow containsString:@"your creator"] || [qLow containsString:@"your developer"] ||
                       [qLow containsString:@"your builder"] || [qLow containsString:@"your founder"] ||
                       [qLow containsString:@"your author"] || [qLow containsString:@"created you"] ||
                       [qLow containsString:@"made you"] || [qLow containsString:@"built you"];

    NSString *rLow = [clean lowercaseString];
    BOOL degenerateForCreator = asksCreator && (![rLow containsString:@"melody"] || clean.length < 28 || kronosIsPoisonedHistoryReply(clean));
    if (degenerateForCreator) {
        return @"Melody Leonard (Melody Leonard Oparaugo) created and built me. He is a Senior Software Engineer and Systems Architect with over 10 years of software engineering experience (@IPMScoutek), specializing in distributed systems, Rust inference engines, and mobile architecture.";
    }

    BOOL asksAboutMelody = [qLow containsString:@"melody"] || [qLow containsString:@"leonard"] ||
                           [qLow containsString:@"about me"] || [qLow containsString:@"who am i"] ||
                           [qLow containsString:@"my profile"] || [qLow containsString:@"my background"];
    if (asksAboutMelody && (clean.length < 36 || kronosIsPoisonedHistoryReply(clean))) {
        return @"Melody Leonard (Melody Leonard Oparaugo) is a Senior Software Engineer and Systems Architect with over 10 years of software engineering experience (@IPMScoutek), and the creator of Kronos. His expertise spans distributed systems, event-driven microservices (Apache Kafka, CQRS, GraphQL, gRPC), Rust systems and Apple Silicon inference engines (kronos-2t), database storage internals (PostgreSQL, MySQL, MongoDB, Neo4j, Redis), and offline-first mobile applications.\n\nGitHub: https://github.com/melodyLeonard\nLinkedIn: https://www.linkedin.com/in/melody-leonard-a362b6192";
    }
    return clean;
}

static inline NSArray<NSString *> *kronosExtractExternalLookupTerms(NSString *query) {
    NSString *normQ = kronosNormalizeApostrophes(query ?: @"");
    NSSet<NSString *> *stops = [NSSet setWithArray:@[
        @"what", @"whats", @"was", @"were", @"is", @"are", @"the", @"a", @"an", @"s",
        @"last", @"latest", @"recent", @"unread", @"message", @"messages", @"chat", @"chats",
        @"on", @"in", @"from", @"to", @"with", @"by", @"about", @"for", @"of", @"and", @"or",
        @"whatsapp", @"whats", @"app", @"imessage", @"sms", @"telegram", @"slack", @"kronos",
        @"said", @"say", @"sent", @"send", @"text", @"texted", @"messaged", @"reply", @"replied",
        @"write", @"wrote", @"written", @"thing", @"thread", @"conversation", @"conversations",
        @"me", @"my", @"our",
        @"did", @"does", @"do", @"have", @"has", @"had", @"today", @"yesterday", @"please",
        @"tell", @"show", @"check", @"read", @"who", @"when", @"how", @"that", @"this",
        @"call", @"called", @"calls", @"calling", @"after", @"since", @"before", @"her", @"his", @"she", @"he", @"him",
        @"they", @"them", @"back", @"yet", @"still", @"responded", @"respond", @"answered", @"answer"
    ]];
    NSMutableCharacterSet *keepSet = [[NSCharacterSet alphanumericCharacterSet] mutableCopy];
    [keepSet addCharactersInString:@"-"];
    NSCharacterSet *splitSet = [keepSet invertedSet];
    NSCharacterSet *trimHyphen = [NSCharacterSet characterSetWithCharactersInString:@"-"];

    NSMutableArray<NSString *> *terms = [NSMutableArray array];
    NSArray<NSString *> *rawTokens = [normQ componentsSeparatedByCharactersInSet:splitSet];
    for (NSString *rawTok in rawTokens) {
        NSString *tok = [rawTok stringByTrimmingCharactersInSet:trimHyphen];
        if (tok.length == 0) continue;
        NSString *low = [tok lowercaseString];
        if ([low hasSuffix:@"s"] && [normQ containsString:[tok stringByAppendingString:@"'s"]]) {
            low = [low substringToIndex:low.length - 1];
        }
        if (low.length >= 2 && ![stops containsObject:low] && ![terms containsObject:low]) {
            [terms addObject:low];
        }
    }
    return terms;
}

static inline NSString *kronosFormatContactDisplayName(NSString *rawTerm) {
    if (rawTerm.length == 0) return @"that contact";
    if ([rawTerm containsString:@"-"]) {
        NSArray<NSString *> *parts = [rawTerm componentsSeparatedByString:@"-"];
        NSMutableArray<NSString *> *formatted = [NSMutableArray arrayWithCapacity:parts.count];
        for (NSUInteger i = 0; i < parts.count; i++) {
            NSString *p = parts[i];
            if (p.length == 0) continue;
            if (i == 0) {
                [formatted addObject:[p capitalizedString]];
            } else {
                [formatted addObject:(p.length <= 2 ? [p lowercaseString] : [p capitalizedString])];
            }
        }
        if (formatted.count > 0) {
            return [formatted componentsJoinedByString:@"-"];
        }
    }
    return [rawTerm capitalizedString];
}

@protocol KronosPasteAttachmentDelegate <NSObject>
- (void)inputTextViewDidPasteImage:(UIImage *)image;
- (void)inputTextViewDidPasteFileData:(NSData *)data filename:(NSString *)filename mime:(NSString *)mime;
@end

@interface KronosInputTextView : UITextView
@property (nonatomic, weak) id<KronosPasteAttachmentDelegate> attachmentPasteDelegate;
@end

@implementation KronosInputTextView
- (BOOL)canPerformAction:(SEL)action withSender:(id)sender {
    if (action == @selector(paste:)) {
        UIPasteboard *pb = [UIPasteboard generalPasteboard];
        if (pb.hasImages || pb.hasURLs || pb.hasStrings || pb.items.count > 0) {
            return YES;
        }
    }
    return [super canPerformAction:action withSender:sender];
}

- (void)paste:(id)sender {
    UIPasteboard *pb = [UIPasteboard generalPasteboard];
    if (pb.hasImages && pb.image != nil) {
        [self.attachmentPasteDelegate inputTextViewDidPasteImage:pb.image];
        return;
    }
    for (NSDictionary<NSString *, id> *item in pb.items) {
        NSData *pdfData = [pb dataForPasteboardType:@"com.adobe.pdf"];
        if (pdfData.length > 0) {
            [self.attachmentPasteDelegate inputTextViewDidPasteFileData:pdfData filename:@"pasted_document.pdf" mime:@"application/pdf"];
            return;
        }
        for (NSString *uti in item.allKeys) {
            if ([uti containsString:@"image"]) {
                id val = item[uti];
                if ([val isKindOfClass:[UIImage class]]) {
                    [self.attachmentPasteDelegate inputTextViewDidPasteImage:(UIImage *)val];
                    return;
                } else if ([val isKindOfClass:[NSData class]]) {
                    UIImage *img = [UIImage imageWithData:(NSData *)val];
                    if (img != nil) {
                        [self.attachmentPasteDelegate inputTextViewDidPasteImage:img];
                        return;
                    }
                }
            }
        }
    }
    if (pb.hasURLs && pb.URL != nil && pb.URL.isFileURL) {
        NSData *fileBytes = [NSData dataWithContentsOfURL:pb.URL];
        if (fileBytes.length > 0) {
            [self.attachmentPasteDelegate inputTextViewDidPasteFileData:fileBytes
                                                               filename:pb.URL.lastPathComponent
                                                                   mime:@"application/octet-stream"];
            return;
        }
    }
    [super paste:sender];
}
@end

@interface KronosConversationStore : NSObject
+ (NSString *)storeFilePath;
+ (NSMutableArray<NSMutableDictionary *> *)loadConversations;
+ (void)saveConversations:(NSArray<NSDictionary *> *)conversations;
+ (NSMutableDictionary *)createBlankConversation;
+ (NSString *)buildMemoryContextForQuery:(NSString *)query
                           conversations:(NSArray<NSDictionary *> *)conversations
                                activeId:(NSString *)activeId;
+ (void)resolveExternalMessageQuery:(NSString *)query
                         completion:(void (^)(BOOL foundVerifiedContext, NSString *verifiedContextOrDirectReply))completion;
@end

@implementation KronosConversationStore
+ (NSString *)storeFilePath {
    return [kronosDocumentsDirectory() stringByAppendingPathComponent:@"kronos_conversations.json"];
}

+ (NSMutableArray<NSMutableDictionary *> *)loadConversations {
    NSData *data = [NSData dataWithContentsOfFile:[self storeFilePath]];
    if (data.length == 0) return [NSMutableArray array];
    NSArray *raw = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingMutableContainers error:nil];
    if (![raw isKindOfClass:[NSArray class]]) return [NSMutableArray array];
    NSMutableArray<NSMutableDictionary *> *out = [NSMutableArray array];
    for (id obj in raw) {
        if ([obj isKindOfClass:[NSDictionary class]]) {
            NSMutableDictionary *m = [(NSDictionary *)obj mutableCopy];
            if ([m[@"messages"] isKindOfClass:[NSArray class]]) {
                m[@"messages"] = [m[@"messages"] mutableCopy];
            } else {
                m[@"messages"] = [NSMutableArray array];
            }
            [out addObject:m];
        }
    }
    return out;
}

+ (void)saveConversations:(NSArray<NSDictionary *> *)conversations {
    if (conversations == nil) return;
    NSData *data = [NSJSONSerialization dataWithJSONObject:conversations options:0 error:nil];
    if (data != nil) {
        [data writeToFile:[self storeFilePath] atomically:YES];
    }
}

+ (NSMutableDictionary *)createBlankConversation {
    NSTimeInterval now = [[NSDate date] timeIntervalSince1970];
    return [@{
        @"id": [[NSUUID UUID] UUIDString],
        @"title": @"New conversation",
        @"updatedAt": @(now),
        @"messages": [NSMutableArray array],
    } mutableCopy];
}

+ (BOOL)isMetaOrCommandUserTurn:(NSString *)txt {
    NSString *low = [[txt ?: @"" lowercaseString] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (kronosIsActionOrComposerCommand(low) || kronosIsExternalMessageLookupQuery(low)) return YES;
    return [low containsString:@"remind me"] || [low containsString:@"what did we talk"];
}

+ (NSString *)buildMemoryContextForQuery:(NSString *)query
                           conversations:(NSArray<NSDictionary *> *)conversations
                                activeId:(NSString *)activeId {
    NSArray<NSString *> *terms = kronosExtractExternalLookupTerms(query);
    NSMutableString *mem = [NSMutableString stringWithString:
        @"[Local Kronos Conversation Memory on This iPhone — Answer directly from these saved Kronos conversations]\n"];
    NSUInteger added = 0;

    for (NSDictionary *conv in conversations) {
        if (activeId.length > 0 && [conv[@"id"] isEqual:activeId]) continue;
        NSArray<NSDictionary *> *msgs = conv[@"messages"];
        if (msgs.count == 0) continue;
        NSString *title = conv[@"title"] ?: @"Conversation";
        NSString *age = kronosRelativeDateString([conv[@"updatedAt"] doubleValue]);

        NSMutableString *snippet = [NSMutableString string];
        BOOL matchedTerm = (terms.count == 0);
        for (NSDictionary *m in msgs) {
            BOOL isUser = [m[@"role"] isEqualToString:@"user"];
            NSString *txt = m[@"displayText"] ?: (m[@"content"] ?: @"");
            if (txt.length == 0) continue;
            if (isUser && [self isMetaOrCommandUserTurn:txt]) continue;
            NSString *role = isUser ? @"User" : @"Kronos";
            NSString *low = [txt lowercaseString];
            for (NSString *t in terms) {
                if ([low containsString:t] || [[title lowercaseString] containsString:t]) {
                    matchedTerm = YES;
                    break;
                }
            }
            NSString *shortTxt = txt.length > 180 ? [[txt substringToIndex:180] stringByAppendingString:@"..."] : txt;
            shortTxt = [shortTxt stringByReplacingOccurrencesOfString:@"\n" withString:@" "];
            [snippet appendFormat:@"  %@: %@\n", role, shortTxt];
        }
        if (matchedTerm && snippet.length > 0) {
            [mem appendFormat:@"- Saved Conversation \"%@\" (%@):\n%@", title, age, snippet];
            added++;
            if (added >= 5) break;
        }
    }

    if (added == 0) {
        [mem appendString:@"- No matching past Kronos sessions were found on this iPhone yet.\n"];
    }
    return mem;
}

typedef NS_ENUM(NSInteger, KronosMessageDirectionIntent) {
    KronosMsgIntentIncomingFromContact = 0,
    KronosMsgIntentOutgoingFromUser = 1,
    KronosMsgIntentThreadOverview = 2,
};

static inline KronosMessageDirectionIntent kronosParseMessageDirectionIntent(NSString *query) {
    NSString *low = [[kronosNormalizeApostrophes(query ?: @"") lowercaseString]
                     stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    // Mirrors parse_message_direction_intent: non-literal asks are questions about the thread.
    low = [[low stringByReplacingOccurrencesOfString:@"whatsapp" withString:@" "] stringByReplacingOccurrencesOfString:@"whats app" withString:@" "];
    NSCharacterSet *nonWord = [[NSCharacterSet alphanumericCharacterSet] invertedSet];
    NSSet<NSString *> *words = [NSSet setWithArray:[low componentsSeparatedByCharactersInSet:nonWord]];
    BOOL namesMessage = [words intersectsSet:[NSSet setWithArray:@[@"message", @"messages", @"text", @"texts", @"sent", @"send", @"say", @"said", @"wrote", @"write", @"reply", @"replied"]]];
    BOOL asksForIt = NO;
    for (NSString *c in @[@"what was", @"what's", @"whats", @"what is", @"what did", @"show", @"read", @"tell me", @"get "]) {
        if ([low containsString:c]) asksForIt = YES;
    }
    BOOL literal = asksForIt && namesMessage && ![low containsString:@" about "];
    for (NSString *c in @[@"last message", @"latest message", @"recent message", @"last text", @"latest text", @"message from ", @"messages from "]) {
        if ([low containsString:c]) literal = YES;
    }
    for (NSString *c in @[@"after my last", @"since my last", @"after my message", @"since my message", @"after i messaged", @"since i messaged", @"after i texted", @"since i texted", @"replied to my", @"reply to my", @"responded to my", @"respond to my"]) {
        if ([low containsString:c]) return KronosMsgIntentOutgoingFromUser;
    }
    if (!literal) return KronosMsgIntentThreadOverview;
    NSArray<NSString *> *outgoingCues = @[
        @"my last message", @"my latest message", @"my recent message",
        @"my message to ", @"my messages to ", @"my reply to ",
        @"message i sent", @"messages i sent", @"message i wrote",
        @"what did i send", @"what did i write", @"what did i text",
        @"what did i message", @"what did i reply", @"what did i say",
        @"what i sent ", @"what i said to ", @"last thing i sent", @"last thing i said"
    ];
    for (NSString *c in outgoingCues) {
        if ([low containsString:c]) return KronosMsgIntentOutgoingFromUser;
    }
    NSArray<NSString *> *incomingCues = @[
        @"to me", @"sent me", @"send me", @"messaged me", @"texted me",
        @"wrote to me", @"wrote me", @"replied to me", @"said to me",
        @"message from ", @"messages from "
    ];
    for (NSString *c in incomingCues) {
        if ([low containsString:c]) return KronosMsgIntentIncomingFromContact;
    }
    NSArray<NSString *> *overviewCues = @[
        @"recent messages with ", @"messages between ", @"conversation with ",
        @"chat history with ", @"summarize my chat", @"summarize my conversation", @"thread with "
    ];
    for (NSString *c in overviewCues) {
        if ([low containsString:c]) return KronosMsgIntentThreadOverview;
    }
    return KronosMsgIntentIncomingFromContact;
}

// Strict contact-attributed and direction-aware lookup for WhatsApp / external messages:
// 1. Inspects genuine imported WhatsApp chat exports (.txt) in KronosWorkspace, distinguishing messages sent BY the contact vs sent BY you.
// 2. Queries the synced local Mac Kronos server (/v1/device/conversations?q=...) over LAN when enabled, respecting incoming vs outgoing intent.
// 3. Never matches Kronos user prompts (like "Open WhatsApp and write hello to Dera") and never fabricates "Hello" or misattributes your outgoing replies to the contact.
+ (void)resolveExternalMessageQuery:(NSString *)query
                         completion:(void (^)(BOOL foundVerifiedContext, NSString *verifiedContextOrDirectReply))completion {
    NSArray<NSString *> *terms = kronosExtractExternalLookupTerms(query);
    KronosMessageDirectionIntent dirIntent = kronosParseMessageDirectionIntent(query);
    NSString *contactDisplay = terms.firstObject.length > 0
        ? kronosFormatContactDisplayName(terms.firstObject)
        : @"that contact";

    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        NSMutableArray<NSString *> *localExportMatches = [NSMutableArray array];
        NSString *ws = kronosWorkspaceDirectory();
        NSArray<NSString *> *files = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:ws error:nil] ?: @[];
        for (NSString *fn in files) {
            NSString *lowFn = [fn lowercaseString];
            if (![lowFn hasSuffix:@".txt"]) continue;
            NSString *full = [ws stringByAppendingPathComponent:fn];
            NSString *body = [NSString stringWithContentsOfFile:full encoding:NSUTF8StringEncoding error:nil];
            if (body.length == 0) continue;
            NSArray<NSString *> *fnWords = kronosSplitAlphaWords(lowFn);
            BOOL fileBelongsToContact = (terms.count > 0);
            for (NSString *t in terms) {
                for (NSString *sw in kronosSplitAlphaWords(t)) {
                    if (![fnWords containsObject:sw]) {
                        fileBelongsToContact = NO;
                        break;
                    }
                }
                if (!fileBelongsToContact) break;
            }
            NSArray<NSString *> *lines = [body componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
            // If filename doesn't explicitly contain the contact name, scan whether any line sender in the export matches the contact.
            if (!fileBelongsToContact && terms.count > 0) {
                for (NSString *probeLine in lines) {
                    NSRange pColon = [probeLine rangeOfString:@": "];
                    if (pColon.location == NSNotFound) continue;
                    NSArray<NSString *> *pWords = kronosSplitAlphaWords([[probeLine substringToIndex:pColon.location] lowercaseString]);
                    BOOL hitAll = YES;
                    for (NSString *sw in kronosSplitAlphaWords(terms.firstObject)) {
                        if (![pWords containsObject:sw]) { hitAll = NO; break; }
                    }
                    if (hitAll) { fileBelongsToContact = YES; break; }
                }
            }
            for (NSString *line in [lines reverseObjectEnumerator]) {
                // Match authentic WhatsApp exported chat lines: "[date, time] Sender: message" or "date, time - Sender: message"
                BOOL looksLikeExport = ([line hasPrefix:@"["] && [line containsString:@"] "] && [line containsString:@": "]) ||
                                       ([line containsString:@" - "] && [line containsString:@": "]);
                if (!looksLikeExport) continue;
                NSRange colon = [line rangeOfString:@": "];
                if (colon.location == NSNotFound) continue;
                NSString *headerPart = [line substringToIndex:colon.location];
                NSString *senderOnly = headerPart;
                NSRange closeBr = [headerPart rangeOfString:@"] "];
                if (closeBr.location != NSNotFound) {
                    senderOnly = [headerPart substringFromIndex:closeBr.location + 2];
                } else {
                    NSRange dashSep = [headerPart rangeOfString:@" - "];
                    if (dashSep.location != NSNotFound) {
                        senderOnly = [headerPart substringFromIndex:dashSep.location + 3];
                    }
                }
                NSArray<NSString *> *senderWords = kronosSplitAlphaWords([senderOnly lowercaseString]);
                BOOL senderIsContact = (terms.count > 0);
                for (NSString *sw in kronosSplitAlphaWords(terms.firstObject ?: @"")) {
                    if (![senderWords containsObject:sw]) {
                        senderIsContact = NO;
                        break;
                    }
                }
                BOOL senderIsOwner = [senderWords containsObject:@"melody"] || [senderWords containsObject:@"you"] ||
                                     (fileBelongsToContact && !senderIsContact);

                BOOL acceptLine = NO;
                if (terms.count == 0) {
                    acceptLine = YES;
                } else if (dirIntent == KronosMsgIntentIncomingFromContact) {
                    acceptLine = senderIsContact;
                } else if (dirIntent == KronosMsgIntentOutgoingFromUser) {
                    acceptLine = fileBelongsToContact && senderIsOwner && !senderIsContact;
                } else {
                    acceptLine = senderIsContact || (fileBelongsToContact && senderIsOwner);
                }

                if (acceptLine) {
                    [localExportMatches addObject:line];
                    if (localExportMatches.count >= 8) break;
                }
            }
        }

        if (localExportMatches.count > 0) {
            if (dirIntent != KronosMsgIntentThreadOverview) {
                NSString *topLine = localExportMatches.firstObject;
                NSRange col = [topLine rangeOfString:@": "];
                NSString *bodyText = (col.location != NSNotFound) ? [topLine substringFromIndex:(col.location + 2)] : topLine;
                NSString *hdr = (dirIntent == KronosMsgIntentOutgoingFromUser)
                    ? [NSString stringWithFormat:@"Latest outgoing message FROM YOU to %@ (WhatsApp export)", contactDisplay]
                    : [NSString stringWithFormat:@"Latest incoming message FROM %@ to you (WhatsApp export)", contactDisplay];
                NSString *directExport = [NSString stringWithFormat:@"%@:\n\n\"%@\"", hdr, bodyText];
                dispatch_async(dispatch_get_main_queue(), ^{
                    completion(NO, directExport);
                });
                return;
            }
            NSString *intentTag = [NSString stringWithFormat:@"Messages with %@", contactDisplay];
            NSMutableString *ctx = [NSMutableString stringWithFormat:
                @"[Verified Imported WhatsApp Chat Export — %@ (newest first)]\n", intentTag];
            for (NSString *l in localExportMatches) {
                [ctx appendFormat:@"- %@\n", l];
            }
            dispatch_async(dispatch_get_main_queue(), ^{
                completion(YES, ctx);
            });
            return;
        }

        // Query synced local Mac Kronos server over Wi-Fi LAN if enabled.
        KronosMobileSettings *settings = [KronosMobileSettings sharedSettings];
        BOOL macReached = NO;
        BOOL macTccBlocked = NO;
        if (settings.macSyncEnabled && settings.macServerURL.length > 0) {
            NSString *base = [settings.macServerURL stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@"/ "]];
            NSString *encQ = [query stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet alphanumericCharacterSet]] ?: @"";
            NSURL *url = [NSURL URLWithString:[NSString stringWithFormat:@"%@/v1/device/conversations?q=%@&tz=%@", base, encQ, [NSTimeZone localTimeZone].name]];
            if (url != nil) {
                NSMutableURLRequest *req = [NSMutableURLRequest requestWithURL:url
                                                                   cachePolicy:NSURLRequestReloadIgnoringLocalCacheData
                                                               timeoutInterval:6.0];
                kronosApplyMacAuth(req);
                dispatch_semaphore_t sem = dispatch_semaphore_create(0);
                __block NSData *respData = nil;
                __block NSInteger statusCode = 0;
                [[[NSURLSession sharedSession] dataTaskWithRequest:req completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
                    if (error == nil && [response isKindOfClass:[NSHTTPURLResponse class]]) {
                        statusCode = [(NSHTTPURLResponse *)response statusCode];
                        respData = data;
                    }
                    dispatch_semaphore_signal(sem);
                }] resume];
                dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(6.5 * NSEC_PER_SEC)));
                if (statusCode == 200 && respData.length > 0) {
                    NSDictionary *json = [NSJSONSerialization JSONObjectWithData:respData options:0 error:nil];
                    if ([json isKindOfClass:[NSDictionary class]]) {
                        macReached = YES;
                        [KronosIosDeviceContextManager sharedManager].macReachable = YES;
                        BOOL verified = [json[@"found_verified_match"] boolValue];
                        macTccBlocked = [json[@"whatsapp_tcc_blocked"] boolValue];
                        NSString *macCtx = json[@"context"];
                        if (verified && [macCtx isKindOfClass:[NSString class]] && macCtx.length > 0) {
                            // 1. Deterministic directional safeguard when sidebar preview is YOUR outgoing reply while user asked for contact's incoming message.
                            if ([macCtx containsString:@"IMPORTANT DIRECTIONALITY CHECK:"] && ![macCtx containsString:@"VERIFIED DIRECT ANSWER ("]) {
                                for (NSString *ln in [macCtx componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]]) {
                                    NSRange dr = [ln rangeOfString:@"IMPORTANT DIRECTIONALITY CHECK: "];
                                    if (dr.location != NSNotFound) {
                                        NSString *note = [[ln substringFromIndex:dr.location + dr.length]
                                                          stringByReplacingOccurrencesOfString:@"`" withString:@""];
                                        NSRange tellIdx = [note rangeOfString:@" Tell the user"];
                                        if (tellIdx.location != NSNotFound) {
                                            note = [note substringToIndex:tellIdx.location];
                                        }
                                        NSString *honestDir = [NSString stringWithFormat:
                                            @"%@\n\nClick %@'s chat open in WhatsApp on your Mac (or export the chat into Kronos Mobile) so I can read %@'s last incoming message to you.",
                                            note, contactDisplay, contactDisplay];
                                        dispatch_async(dispatch_get_main_queue(), ^{
                                            completion(NO, honestDir);
                                        });
                                        return;
                                    }
                                }
                            }
                            // 2. Deterministic direct rendering when Mac perception produced a verified direction-matched answer.
                            // Thread rows are rendered as chat bubbles by the caller.
                            if ([macCtx containsString:@"VERIFIED DIRECT ANSWER ("] && dirIntent != KronosMsgIntentThreadOverview && ![macCtx containsString:@"\n  - ["]) {
                                for (NSString *ln in [macCtx componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]]) {
                                    NSRange vr = [ln rangeOfString:@"VERIFIED DIRECT ANSWER ("];
                                    if (vr.location != NSNotFound) {
                                        NSString *tail = [[ln substringFromIndex:vr.location + vr.length]
                                                          stringByReplacingOccurrencesOfString:@"`" withString:@""];
                                        NSRange closeParen = [tail rangeOfString:@"): "];
                                        if (closeParen.location != NSNotFound) {
                                            NSString *headerDesc = [tail substringToIndex:closeParen.location];
                                            NSString *msgAndNote = [tail substringFromIndex:closeParen.location + 3];
                                            NSRange ctxNote = [msgAndNote rangeOfString:@" (Context note: "];
                                            NSString *cleanMsg = (ctxNote.location != NSNotFound)
                                                ? [msgAndNote substringToIndex:ctxNote.location]
                                                : msgAndNote;
                                            NSString *direct = [NSString stringWithFormat:@"%@:\n\n%@", headerDesc, cleanMsg];
                                            dispatch_async(dispatch_get_main_queue(), ^{
                                                completion(NO, direct);
                                            });
                                            return;
                                        }
                                    }
                                }
                            }
                            dispatch_async(dispatch_get_main_queue(), ^{
                                completion(YES, macCtx);
                            });
                            return;
                        }
                    }
                }
            }
        }

        // Truthful deterministic report when no verified message for that contact is accessible yet.
        NSMutableString *honest = [NSMutableString string];
        if (terms.count > 0) {
            [honest appendFormat:@"I checked your local context and could not find a verified WhatsApp message for %@ yet (so I will not guess or attribute another contact's message).\n\n", contactDisplay];
        } else {
            [honest appendString:@"I checked your local context and could not find verified WhatsApp messages yet.\n\n"];
        }
        if (macReached) {
            if (macTccBlocked) {
                [honest setString:@"macOS is blocking Kronos from reading WhatsApp on your Mac. Open System Settings, Privacy & Security, Full Disk Access. Remove kronos_device_bridge, add it again from ~/.kronos/bin, and turn it on. Then restart the bridge."];
            } else {
                [honest appendFormat:@"%@ was not found in the visible WhatsApp window on your synced Mac. You can open %@'s thread on your Mac or export/screenshot the chat into Kronos Mobile.", contactDisplay, contactDisplay];
            }
        } else {
            [honest appendFormat:@"iOS blocks WhatsApp's data on iPhone, so Kronos reads it through your Mac, and it could not reach your Mac at %@. To let Kronos read %@'s real messages:\n", settings.macServerURL, contactDisplay];
            [honest appendString:@"1. Start kronos_device_bridge on your Mac, on the same Wi-Fi, or\n"];
            [honest appendFormat:@"2. In WhatsApp on iPhone, open %@ -> Export Chat (Without Media) or paste a chat screenshot into Kronos Mobile.", contactDisplay];
        }
        dispatch_async(dispatch_get_main_queue(), ^{
            completion(NO, honest);
        });
    });
}
@end

#endif
