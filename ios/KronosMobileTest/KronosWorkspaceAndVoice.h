#ifndef KRONOS_WORKSPACE_AND_VOICE_H
#define KRONOS_WORKSPACE_AND_VOICE_H

#import <UIKit/UIKit.h>
#import <WebKit/WebKit.h>
#import <Speech/Speech.h>
#import <AVFoundation/AVFoundation.h>
#import "KronosThemeAndModels.h"

// Live Web & File Preview Controller for files built by Kronos on iOS.
@interface KronosFilePreviewViewController : UIViewController
@property (nonatomic, copy) NSString *filePath;
@property (nonatomic, strong) WKWebView *webView;
@property (nonatomic, strong) UITextView *codeView;
@property (nonatomic, strong) UISegmentedControl *modeControl;
@end

@implementation KronosFilePreviewViewController
- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = kronosBgColor();
    NSString *fname = self.filePath.lastPathComponent ?: @"File";
    self.title = fname;

    UIBarButtonItem *closeItem = [[UIBarButtonItem alloc] initWithTitle:@"Done"
                                                                  style:UIBarButtonItemStyleDone
                                                                 target:self
                                                                 action:@selector(closeTapped)];
    UIBarButtonItem *shareItem = [[UIBarButtonItem alloc] initWithImage:[UIImage systemImageNamed:@"square.and.arrow.up"]
                                                                  style:UIBarButtonItemStylePlain
                                                                 target:self
                                                                 action:@selector(shareTapped)];
    self.navigationItem.rightBarButtonItem = closeItem;
    self.navigationItem.leftBarButtonItem = shareItem;

    BOOL isHtml = [[fname lowercaseString] hasSuffix:@".html"] || [[fname lowercaseString] hasSuffix:@".htm"];
    UILayoutGuide *safe = self.view.safeAreaLayoutGuide;

    if (isHtml) {
        self.modeControl = [[UISegmentedControl alloc] initWithItems:@[@"Preview", @"Source"]];
        self.modeControl.selectedSegmentIndex = 0;
        [self.modeControl addTarget:self action:@selector(modeChanged:) forControlEvents:UIControlEventValueChanged];
        self.navigationItem.titleView = self.modeControl;

        self.webView = [[WKWebView alloc] initWithFrame:CGRectZero];
        self.webView.translatesAutoresizingMaskIntoConstraints = NO;
        [self.view addSubview:self.webView];

        NSURL *fileURL = [NSURL fileURLWithPath:self.filePath];
        NSURL *dirURL = [fileURL URLByDeletingLastPathComponent];
        [self.webView loadFileURL:fileURL allowingReadAccessToURL:dirURL];

        [NSLayoutConstraint activateConstraints:@[
            [self.webView.topAnchor constraintEqualToAnchor:safe.topAnchor],
            [self.webView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
            [self.webView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
            [self.webView.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor],
        ]];
    }

    self.codeView = [[UITextView alloc] init];
    self.codeView.translatesAutoresizingMaskIntoConstraints = NO;
    self.codeView.backgroundColor = kronosPanelColor();
    self.codeView.textColor = kronosTextColor();
    self.codeView.font = [UIFont monospacedSystemFontOfSize:13.0 weight:UIFontWeightRegular];
    self.codeView.editable = NO;
    self.codeView.textContainerInset = UIEdgeInsetsMake(14, 14, 14, 14);
    NSString *txt = [NSString stringWithContentsOfFile:self.filePath encoding:NSUTF8StringEncoding error:nil];
    self.codeView.text = txt ?: @"(Binary or non-UTF8 file)";
    self.codeView.hidden = isHtml;
    [self.view addSubview:self.codeView];

    [NSLayoutConstraint activateConstraints:@[
        [self.codeView.topAnchor constraintEqualToAnchor:safe.topAnchor],
        [self.codeView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.codeView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.codeView.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor],
    ]];
}

- (void)modeChanged:(UISegmentedControl *)seg {
    BOOL showPreview = (seg.selectedSegmentIndex == 0);
    self.webView.hidden = !showPreview;
    self.codeView.hidden = showPreview;
}

- (void)shareTapped {
    if (self.filePath.length == 0) return;
    NSURL *url = [NSURL fileURLWithPath:self.filePath];
    UIActivityViewController *avc = [[UIActivityViewController alloc] initWithActivityItems:@[url] applicationActivities:nil];
    [self presentViewController:avc animated:YES completion:nil];
}

- (void)closeTapped {
    [self dismissViewControllerAnimated:YES completion:nil];
}
@end

@interface KronosWorkspaceListViewController : UIViewController <UITableViewDataSource, UITableViewDelegate>
@property (nonatomic, strong) UITableView *tableView;
@property (nonatomic, strong) NSArray<NSString *> *files;
@end

@implementation KronosWorkspaceListViewController
- (void)viewDidLoad {
    [super viewDidLoad];
    self.title = @"Workspace Files";
    self.view.backgroundColor = kronosBgColor();
    self.navigationItem.rightBarButtonItem = [[UIBarButtonItem alloc] initWithTitle:@"Done"
                                                                              style:UIBarButtonItemStyleDone
                                                                             target:self
                                                                             action:@selector(closeTapped)];
    self.tableView = [[UITableView alloc] initWithFrame:CGRectZero style:UITableViewStyleInsetGrouped];
    self.tableView.translatesAutoresizingMaskIntoConstraints = NO;
    self.tableView.backgroundColor = kronosBgColor();
    self.tableView.dataSource = self;
    self.tableView.delegate = self;
    [self.view addSubview:self.tableView];

    [NSLayoutConstraint activateConstraints:@[
        [self.tableView.topAnchor constraintEqualToAnchor:self.view.topAnchor],
        [self.tableView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.tableView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.tableView.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor],
    ]];
    [self reloadFiles];
}

- (void)reloadFiles {
    NSString *ws = kronosWorkspaceDirectory();
    NSArray<NSString *> *raw = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:ws error:nil] ?: @[];
    NSMutableArray<NSString *> *clean = [NSMutableArray array];
    for (NSString *f in raw) {
        if (![f hasPrefix:@"."]) {
            [clean addObject:f];
        }
    }
    [clean sortUsingSelector:@selector(localizedCaseInsensitiveCompare:)];
    self.files = clean;
    [self.tableView reloadData];
}

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    return MAX((NSInteger)self.files.count, 1);
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:@"FileCell"];
    if (cell == nil) {
        cell = [[UITableViewCell alloc] initWithStyle:UITableViewCellStyleSubtitle reuseIdentifier:@"FileCell"];
    }
    if (self.files.count == 0) {
        cell.textLabel.text = @"No workspace files yet";
        cell.textLabel.textColor = kronosMutedColor();
        cell.detailTextLabel.text = @"Attach a file or ask Kronos to create a page, note, or script.";
        cell.detailTextLabel.textColor = kronosMutedColor();
        cell.imageView.image = [UIImage systemImageNamed:@"folder"];
        cell.imageView.tintColor = kronosMutedColor();
        cell.selectionStyle = UITableViewCellSelectionStyleNone;
        cell.accessoryType = UITableViewCellAccessoryNone;
        return cell;
    }
    NSString *fname = self.files[(NSUInteger)indexPath.row];
    NSString *full = [kronosWorkspaceDirectory() stringByAppendingPathComponent:fname];
    NSDictionary *attrs = [[NSFileManager defaultManager] attributesOfItemAtPath:full error:nil];
    NSUInteger sz = [attrs[NSFileSize] unsignedIntegerValue];
    BOOL isHtml = [[fname lowercaseString] hasSuffix:@".html"] || [[fname lowercaseString] hasSuffix:@".htm"];

    cell.textLabel.text = fname;
    cell.textLabel.textColor = kronosTextColor();
    cell.detailTextLabel.text = isHtml
        ? [NSString stringWithFormat:@"%@ · Tap for live preview", kronosFormatBytes(sz)]
        : kronosFormatBytes(sz);
    cell.detailTextLabel.textColor = kronosMutedColor();
    cell.imageView.image = [UIImage systemImageNamed:isHtml ? @"safari" : @"doc.text"];
    cell.imageView.tintColor = kronosAccentColor();
    cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
    return cell;
}

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
    if (self.files.count == 0) return;
    NSString *fname = self.files[(NSUInteger)indexPath.row];
    KronosFilePreviewViewController *pvc = [[KronosFilePreviewViewController alloc] init];
    pvc.filePath = [kronosWorkspaceDirectory() stringByAppendingPathComponent:fname];
    [self.navigationController pushViewController:pvc animated:YES];
}

- (void)closeTapped {
    [self dismissViewControllerAnimated:YES completion:nil];
}
@end

// Native iOS Settings & Device Context Sheet (matching Kronos Web settings with native mobile ergonomics).
@protocol KronosSettingsViewControllerDelegate <NSObject>
- (void)settingsViewControllerDidUpdateSettings;
@end

@interface KronosSettingsViewController : UIViewController <UITableViewDataSource, UITableViewDelegate>
@property (nonatomic, weak) id<KronosSettingsViewControllerDelegate> delegate;
@property (nonatomic, strong) UITableView *tableView;
@end

@implementation KronosSettingsViewController
- (void)viewDidLoad {
    [super viewDidLoad];
    self.title = @"Settings";
    self.view.backgroundColor = kronosBgColor();
    self.navigationItem.rightBarButtonItem = [[UIBarButtonItem alloc] initWithTitle:@"Done"
                                                                              style:UIBarButtonItemStyleDone
                                                                             target:self
                                                                             action:@selector(closeTapped)];
    self.navigationItem.rightBarButtonItem.tintColor = kronosAccentColor();

    self.tableView = [[UITableView alloc] initWithFrame:CGRectZero style:UITableViewStyleInsetGrouped];
    self.tableView.translatesAutoresizingMaskIntoConstraints = NO;
    self.tableView.backgroundColor = kronosBgColor();
    self.tableView.dataSource = self;
    self.tableView.delegate = self;
    [self.view addSubview:self.tableView];

    [NSLayoutConstraint activateConstraints:@[
        [self.tableView.topAnchor constraintEqualToAnchor:self.view.topAnchor],
        [self.tableView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.tableView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.tableView.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor],
    ]];

    if ([KronosMobileSettings sharedSettings].developDeviceContext) {
        __weak typeof(self) weakSelf = self;
        [[KronosIosDeviceContextManager sharedManager] refreshDeviceContextInteractive:NO completion:^{
            [weakSelf.tableView reloadData];
            [weakSelf.delegate settingsViewControllerDidUpdateSettings];
        }];
    }
}

- (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView {
    return 4;
}

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    (void)tableView;
    if (section == 0) return 2; // Develop Device Context + On-Device Index refresh
    if (section == 1) return 1; // Allow Phone Use
    if (section == 2) return 3; // Sync with Local Mac + Use Mac 14B Engine + Mac Server Address
    return 2;                   // Spoken Voice Replies + Auto-Send Voice After Pause
}

- (NSString *)tableView:(UITableView *)tableView titleForHeaderInSection:(NSInteger)section {
    (void)tableView;
    if (section == 0) return @"DEVICE CONTEXT & OWNERSHIP";
    if (section == 1) return @"PHONE USE & ACTIONS";
    if (section == 2) return @"LOCAL MAC COMPANION";
    return @"VOICE & SPEECH";
}

- (NSString *)tableView:(UITableView *)tableView titleForFooterInSection:(NSInteger)section {
    (void)tableView;
    if (section == 0) {
        return @"Indexes your Contacts, Photo Library, Calendar, and Workspace files locally on this iPhone. All data stays on your device under your ownership.";
    }
    if (section == 1) {
        return @"Lets Kronos open apps, pre-fill WhatsApp message drafts, and prepare LinkedIn posts with your photos. Kronos always asks you to review before sending or publishing.";
    }
    if (section == 2) {
        return @"Connects over local Wi-Fi to your Mac Kronos 14B server for desktop context and high-accuracy local reasoning, falling back automatically to the on-device Qwen2.5-1.5B-Instruct Metal engine when offline.";
    }
    return @"Controls spoken replies and automatic submission when you pause speaking.";
}

- (UISwitch *)makeSwitchWithTag:(NSInteger)tag isOn:(BOOL)isOn {
    UISwitch *sw = [[UISwitch alloc] init];
    sw.onTintColor = kronosAccentColor();
    sw.tag = tag;
    sw.on = isOn;
    [sw addTarget:self action:@selector(switchToggled:) forControlEvents:UIControlEventValueChanged];
    return sw;
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    NSString *cellId = (indexPath.section == 0 && indexPath.row == 1) || (indexPath.section == 2 && indexPath.row == 2)
        ? @"SettingsDetailCell"
        : @"SettingsSwitchCell";
    UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:cellId];
    if (cell == nil) {
        cell = [[UITableViewCell alloc] initWithStyle:UITableViewCellStyleSubtitle reuseIdentifier:cellId];
    }
    cell.backgroundColor = kronosPanelColor();
    cell.textLabel.textColor = kronosTextColor();
    cell.textLabel.font = [UIFont systemFontOfSize:15.5 weight:UIFontWeightMedium];
    cell.detailTextLabel.textColor = kronosMutedColor();
    cell.detailTextLabel.font = [UIFont systemFontOfSize:12.5 weight:UIFontWeightRegular];
    cell.imageView.tintColor = kronosAccentColor();
    cell.accessoryView = nil;
    cell.accessoryType = UITableViewCellAccessoryNone;
    cell.selectionStyle = UITableViewCellSelectionStyleNone;

    KronosMobileSettings *s = [KronosMobileSettings sharedSettings];
    KronosIosDeviceContextManager *mgr = [KronosIosDeviceContextManager sharedManager];

    if (indexPath.section == 0 && indexPath.row == 0) {
        cell.textLabel.text = @"Develop Device Context";
        cell.detailTextLabel.text = @"Local Contacts, Photos, Calendar, and Files";
        cell.imageView.image = [UIImage systemImageNamed:@"sparkles.rectangle.stack"];
        cell.accessoryView = [self makeSwitchWithTag:100 isOn:s.developDeviceContext];
    } else if (indexPath.section == 0 && indexPath.row == 1) {
        cell.textLabel.text = @"Refresh Device Context";
        cell.detailTextLabel.text = [mgr indexSummaryText];
        cell.imageView.image = [UIImage systemImageNamed:@"arrow.clockwise.circle"];
        cell.selectionStyle = UITableViewCellSelectionStyleDefault;
        if (mgr.isRefreshing) {
            UIActivityIndicatorView *spin = [[UIActivityIndicatorView alloc] initWithActivityIndicatorStyle:UIActivityIndicatorViewStyleMedium];
            [spin startAnimating];
            cell.accessoryView = spin;
        } else {
            cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
        }
    } else if (indexPath.section == 1 && indexPath.row == 0) {
        cell.textLabel.text = @"Allow Phone Use";
        cell.detailTextLabel.text = @"Open apps, stage WhatsApp & LinkedIn drafts";
        cell.imageView.image = [UIImage systemImageNamed:@"iphone.radiowaves.left.and.right"];
        cell.accessoryView = [self makeSwitchWithTag:101 isOn:s.allowPhoneUse];
    } else if (indexPath.section == 2 && indexPath.row == 0) {
        cell.textLabel.text = @"Sync with Local Mac";
        cell.detailTextLabel.text = @"Bridge desktop workspace & chat context";
        cell.imageView.image = [UIImage systemImageNamed:@"laptopcomputer.and.iphone"];
        cell.accessoryView = [self makeSwitchWithTag:102 isOn:s.macSyncEnabled];
    } else if (indexPath.section == 2 && indexPath.row == 1) {
        cell.textLabel.text = @"Use Mac 14B Engine on Wi-Fi";
        cell.detailTextLabel.text = @"High-accuracy local 14B model when on LAN";
        cell.imageView.image = [UIImage systemImageNamed:@"cpu"];
        cell.accessoryView = [self makeSwitchWithTag:105 isOn:s.preferMacEngineOnLan];
    } else if (indexPath.section == 2 && indexPath.row == 2) {
        cell.textLabel.text = @"Mac Server Address";
        cell.detailTextLabel.text = [mgr macSyncStatusText];
        cell.imageView.image = [UIImage systemImageNamed:@"network"];
        cell.selectionStyle = UITableViewCellSelectionStyleDefault;
        cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
    } else if (indexPath.section == 3 && indexPath.row == 0) {
        cell.textLabel.text = @"Spoken Voice Replies";
        cell.detailTextLabel.text = @"Read Kronos responses aloud automatically";
        cell.imageView.image = [UIImage systemImageNamed:@"speaker.wave.2.fill"];
        cell.accessoryView = [self makeSwitchWithTag:103 isOn:s.spokenRepliesEnabled];
    } else if (indexPath.section == 3 && indexPath.row == 1) {
        cell.textLabel.text = @"Auto-Send Voice After Pause";
        cell.detailTextLabel.text = @"Submit spoken prompt after a brief pause";
        cell.imageView.image = [UIImage systemImageNamed:@"mic.badge.plus"];
        cell.accessoryView = [self makeSwitchWithTag:104 isOn:s.autoSendVoiceAfterPause];
    }
    return cell;
}

- (void)switchToggled:(UISwitch *)sw {
    KronosMobileSettings *s = [KronosMobileSettings sharedSettings];
    if (sw.tag == 100) {
        s.developDeviceContext = sw.isOn;
        [s save];
        if (sw.isOn) {
            __weak typeof(self) weakSelf = self;
            [[KronosIosDeviceContextManager sharedManager] refreshDeviceContextInteractive:YES completion:^{
                [weakSelf.tableView reloadData];
                [weakSelf.delegate settingsViewControllerDidUpdateSettings];
            }];
        }
    } else if (sw.tag == 101) {
        s.allowPhoneUse = sw.isOn;
        [s save];
    } else if (sw.tag == 102) {
        s.macSyncEnabled = sw.isOn;
        [s save];
        if (sw.isOn) {
            __weak typeof(self) weakSelf = self;
            [[KronosIosDeviceContextManager sharedManager] refreshDeviceContextInteractive:NO completion:^{
                [weakSelf.tableView reloadData];
                [weakSelf.delegate settingsViewControllerDidUpdateSettings];
            }];
        }
    } else if (sw.tag == 105) {
        s.preferMacEngineOnLan = sw.isOn;
        [s save];
    } else if (sw.tag == 103) {
        s.spokenRepliesEnabled = sw.isOn;
        [s save];
    } else if (sw.tag == 104) {
        s.autoSendVoiceAfterPause = sw.isOn;
        [s save];
    }
    [self.tableView reloadData];
    [self.delegate settingsViewControllerDidUpdateSettings];
}

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
    if (indexPath.section == 0 && indexPath.row == 1) {
        if (![KronosMobileSettings sharedSettings].developDeviceContext) {
            [KronosMobileSettings sharedSettings].developDeviceContext = YES;
            [[KronosMobileSettings sharedSettings] save];
        }
        [self.tableView reloadData];
        __weak typeof(self) weakSelf = self;
        [[KronosIosDeviceContextManager sharedManager] refreshDeviceContextInteractive:YES completion:^{
            [weakSelf.tableView reloadData];
            [weakSelf.delegate settingsViewControllerDidUpdateSettings];
        }];
    } else if (indexPath.section == 2 && indexPath.row == 2) {
        KronosMobileSettings *s = [KronosMobileSettings sharedSettings];
        UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Mac Server Address"
                                                                       message:@"Enter the local Wi-Fi URL of kronos_server running on your Mac (for example http://192.168.1.70:18080)."
                                                                preferredStyle:UIAlertControllerStyleAlert];
        [alert addTextFieldWithConfigurationHandler:^(UITextField *tf) {
            tf.text = s.macServerURL;
            tf.placeholder = @"http://192.168.1.70:18080";
            tf.keyboardType = UIKeyboardTypeURL;
            tf.autocapitalizationType = UITextAutocapitalizationTypeNone;
            tf.autocorrectionType = UITextAutocorrectionTypeNo;
        }];
        [alert addAction:[UIAlertAction actionWithTitle:@"Cancel" style:UIAlertActionStyleCancel handler:nil]];
        __weak typeof(self) weakSelf = self;
        [alert addAction:[UIAlertAction actionWithTitle:@"Save & Test" style:UIAlertActionStyleDefault handler:^(UIAlertAction *a) {
            (void)a;
            NSString *val = [alert.textFields.firstObject.text stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
            if (val.length > 0) {
                if (![val hasPrefix:@"http://"] && ![val hasPrefix:@"https://"]) {
                    val = [@"http://" stringByAppendingString:val];
                }
                s.macServerURL = val;
                s.macSyncEnabled = YES;
                [s save];
                [weakSelf.tableView reloadData];
                [[KronosIosDeviceContextManager sharedManager] refreshDeviceContextInteractive:NO completion:^{
                    [weakSelf.tableView reloadData];
                    [weakSelf.delegate settingsViewControllerDidUpdateSettings];
                }];
            }
        }]];
        [self presentViewController:alert animated:YES completion:nil];
    }
}

- (void)closeTapped {
    [self dismissViewControllerAnimated:YES completion:nil];
}
@end

// Helpers for Agentic WhatsApp Message Drafting & LinkedIn Post Composer Workflows.
static inline NSString *kronosCleanExtractedSpan(NSString *raw) {
    if (raw.length == 0) return @"";
    NSCharacterSet *trimSet = [NSCharacterSet characterSetWithCharactersInString:@" \t\n\r\"'.,;?"];
    NSString *s = [raw stringByTrimmingCharactersInSet:trimSet];
    NSArray<NSString *> *trailers = @[@" on whatsapp", @" in whatsapp", @" via whatsapp", @" using whatsapp",
                                      @" on whats app", @" in whats app", @" please", @" right now", @" now"];
    BOOL changed = YES;
    while (changed && s.length > 0) {
        changed = NO;
        NSString *low = [s lowercaseString];
        for (NSString *tr in trailers) {
            if ([low hasSuffix:tr]) {
                s = [[s substringToIndex:s.length - tr.length] stringByTrimmingCharactersInSet:trimSet];
                changed = YES;
                break;
            }
        }
    }
    return s;
}

static inline BOOL kronosIsGenericMessagePlaceholder(NSString *span) {
    NSString *low = [kronosCleanExtractedSpan(span) lowercaseString];
    if (low.length == 0) return YES;
    NSSet<NSString *> *generics = [NSSet setWithArray:@[
        @"a message", @"message", @"a text", @"text", @"a note", @"note",
        @"something", @"anything", @"a whatsapp message", @"whatsapp message",
        @"a quick message", @"quick message", @"a short message", @"a chat",
        @"a reply", @"reply", @"a dm", @"dm", @"it", @"this"
    ]];
    return [generics containsObject:low];
}

static inline BOOL kronosParseWhatsAppDraftIntent(NSString *userText,
                                                  NSString **outRecipient,
                                                  NSString **outExplicitMessage,
                                                  BOOL *outIsTopicInstruction,
                                                  BOOL *outNeedsClarification,
                                                  BOOL *outNeedsRecipient) {
    // Questions or lookup queries ("What was Dera's last message on WhatsApp", "What was the last message Sandra sent on WhatsApp") must NEVER trigger message drafting.
    if (kronosIsInterrogativeOrLookupText(userText)) return NO;

    NSString *normOrig = [kronosNormalizeApostrophes(userText ?: @"") stringByReplacingOccurrencesOfString:@"Whats App" withString:@"WhatsApp"];
    normOrig = [normOrig stringByReplacingOccurrencesOfString:@"whats app" withString:@"whatsapp"];
    normOrig = [normOrig stringByReplacingOccurrencesOfString:@"Whats app" withString:@"WhatsApp"];
    NSString *trimmed = [normOrig stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSString *low = [trimmed lowercaseString];

    if (![low containsString:@"whatsapp"]) return NO;

    NSArray<NSString *> *imperativeStarts = @[
        @"open whatsapp", @"launch whatsapp", @"start whatsapp",
        @"write ", @"send ", @"draft ", @"message ", @"text ", @"tell ", @"ask ", @"say ", @"reply ",
        @"please open whatsapp", @"please write ", @"please send ", @"please message ", @"please text ",
        @"can you open whatsapp", @"can you write ", @"can you send ", @"can you message ",
        @"could you open whatsapp", @"could you write ", @"could you send "
    ];
    BOOL isImperative = NO;
    for (NSString *imp in imperativeStarts) {
        if ([low hasPrefix:imp]) {
            isImperative = YES;
            break;
        }
    }
    if (!isImperative) return NO;

    NSArray<NSString *> *verbs = @[@"write ", @"send ", @"message ", @"text ", @"tell ", @"ask ", @"say ", @"draft ", @"type ", @"reply "];
    BOOL hasVerb = NO;
    for (NSString *v in verbs) {
        if ([low containsString:v]) {
            hasVerb = YES;
            break;
        }
    }
    if (!hasVerb) return NO;

    NSString *recipient = nil;
    NSString *explicitMsg = nil;
    BOOL isTopic = NO;

    // Pattern 1: "... (write|send|draft|type|say) <mid> to <afterTo>"
    NSArray<NSString *> *toVerbs = @[@"write ", @"send ", @"draft ", @"type ", @"say "];
    for (NSString *tv in toVerbs) {
        NSRange vr = [low rangeOfString:tv];
        if (vr.location != NSNotFound) {
            NSUInteger afterVerbIdx = vr.location + vr.length;
            NSString *restOrig = [trimmed substringFromIndex:afterVerbIdx];
            NSString *restLow = [low substringFromIndex:afterVerbIdx];
            NSRange toRange = [restLow rangeOfString:@" to "];
            if (toRange.location != NSNotFound) {
                NSString *mid = kronosCleanExtractedSpan([restOrig substringToIndex:toRange.location]);
                NSString *afterTo = [restOrig substringFromIndex:toRange.location + 4];
                NSString *afterToLow = [afterTo lowercaseString];

                NSArray<NSArray<NSString *> *> *seps = @[
                    @[@" saying ", @"0"],
                    @[@": ", @"0"],
                    @[@" that ", @"0"],
                    @[@" about ", @"1"],
                    @[@" asking ", @"1"],
                    @[@" to ask ", @"1"],
                    @[@" to tell ", @"1"],
                ];
                NSString *tailBody = nil;
                for (NSArray<NSString *> *sp in seps) {
                    NSRange sr = [afterToLow rangeOfString:sp[0]];
                    if (sr.location != NSNotFound) {
                        recipient = kronosCleanExtractedSpan([afterTo substringToIndex:sr.location]);
                        tailBody = kronosCleanExtractedSpan([afterTo substringFromIndex:sr.location + sp[0].length]);
                        isTopic = [sp[1] isEqualToString:@"1"];
                        break;
                    }
                }
                if (recipient == nil) {
                    recipient = kronosCleanExtractedSpan(afterTo);
                }
                if (tailBody.length > 0) {
                    explicitMsg = tailBody;
                } else if (!kronosIsGenericMessagePlaceholder(mid)) {
                    explicitMsg = mid;
                }
                break;
            }
        }
    }

    // Pattern 2: "... (message|text|tell|ask) <Recipient> [body]" — ONLY when verb is at start or preceded by "and " / "to " / "please " / "can you ".
    if (recipient.length == 0) {
        NSArray<NSString *> *directVerbs = @[@"message ", @"text ", @"tell ", @"ask "];
        for (NSString *dv in directVerbs) {
            NSRange vr = [low rangeOfString:dv];
            if (vr.location != NSNotFound) {
                BOOL validVerbBoundary = (vr.location == 0);
                if (!validVerbBoundary) {
                    NSString *before = [low substringToIndex:vr.location];
                    validVerbBoundary = [before hasSuffix:@"and "] || [before hasSuffix:@"to "] ||
                                        [before hasSuffix:@"please "] || [before hasSuffix:@"can you "] ||
                                        [before hasSuffix:@"could you "];
                }
                if (!validVerbBoundary) continue;

                NSString *afterOrig = kronosCleanExtractedSpan([trimmed substringFromIndex:vr.location + vr.length]);
                if (afterOrig.length == 0) continue;
                NSArray<NSString *> *words = [afterOrig componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
                if (words.count == 1) {
                    recipient = kronosCleanExtractedSpan(words[0]);
                } else if (words.count >= 2) {
                    recipient = kronosCleanExtractedSpan(words[0]);
                    NSString *rem = kronosCleanExtractedSpan([[words subarrayWithRange:NSMakeRange(1, words.count - 1)] componentsJoinedByString:@" "]);
                    if (rem.length > 0 && ![[rem lowercaseString] isEqualToString:@"on whatsapp"]) {
                        explicitMsg = rem;
                        isTopic = ([dv isEqualToString:@"ask "] || [[rem lowercaseString] hasPrefix:@"about "] || [[rem lowercaseString] hasPrefix:@"if "]);
                    }
                }
                break;
            }
        }
    }

    NSSet<NSString *> *invalidRecipients = [NSSet setWithArray:@[
        @"", @"whatsapp", @"on", @"in", @"to", @"from", @"with", @"the", @"a", @"an", @"me", @"my", @"message", @"messages", @"sent", @"last"
    ]];
    NSString *recipLow = [recipient ?: @"" lowercaseString];
    if ([invalidRecipients containsObject:recipLow]) {
        BOOL explicitNoRecipDraft = [low containsString:@"write a message"] || [low containsString:@"send a message"] ||
                                    [low containsString:@"draft a message"];
        if (explicitNoRecipDraft) {
            if (outNeedsRecipient) *outNeedsRecipient = YES;
            if (outNeedsClarification) *outNeedsClarification = YES;
            return YES;
        }
        return NO;
    }

    if ([[recipient lowercaseString] isEqualToString:recipient] && recipient.length > 0) {
        recipient = [recipient capitalizedString];
    }

    if (outRecipient) *outRecipient = recipient;
    if (outNeedsRecipient) *outNeedsRecipient = NO;
    if (explicitMsg.length > 0) {
        if (!isTopic && explicitMsg.length > 0) {
            NSString *first = [[explicitMsg substringToIndex:1] uppercaseString];
            explicitMsg = [first stringByAppendingString:[explicitMsg substringFromIndex:1]];
        }
        if (outExplicitMessage) *outExplicitMessage = explicitMsg;
        if (outIsTopicInstruction) *outIsTopicInstruction = isTopic;
        if (outNeedsClarification) *outNeedsClarification = NO;
    } else {
        if (outExplicitMessage) *outExplicitMessage = nil;
        if (outIsTopicInstruction) *outIsTopicInstruction = NO;
        if (outNeedsClarification) *outNeedsClarification = YES;
    }
    return YES;
}

static inline BOOL kronosIsWhatsAppDraftInstructionStyle(NSString *replyText) {
    NSString *low = [[replyText ?: @"" lowercaseString] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSArray<NSString *> *prefixes = @[
        @"a ", @"an ", @"ask ", @"asking ", @"tell ", @"telling ", @"say ", @"saying ",
        @"wish ", @"wishing ", @"remind ", @"reminding ", @"invite ", @"inviting ",
        @"check in", @"checking in", @"thank ", @"thanking ", @"congratulate ",
        @"friendly ", @"casual ", @"formal ", @"professional ", @"polite ", @"short ",
        @"quick ", @"warm ", @"talk about ", @"about ", @"something about "
    ];
    for (NSString *p in prefixes) {
        if ([low hasPrefix:p]) return YES;
    }
    return NO;
}

static inline void kronosOpenWhatsAppDraftForRecipient(NSString *recipient,
                                                       NSString *messageText,
                                                       void (^completion)(BOOL opened, NSString *displayRecipient, BOOL matchedContactPhone, NSURL *draftURL)) {
    NSString *cleanMsg = [messageText ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    [UIPasteboard generalPasteboard].string = cleanMsg;

    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        NSString *matchedName = nil;
        NSString *digits = [[KronosIosDeviceContextManager sharedManager] resolveContactPhoneDigitsForName:recipient
                                                                                            outMatchedName:&matchedName];
        NSString *display = (matchedName.length > 0) ? matchedName : (recipient.length > 0 ? recipient : @"recipient");
        NSMutableCharacterSet *allowed = [[NSCharacterSet URLQueryAllowedCharacterSet] mutableCopy];
        [allowed removeCharactersInString:@"&+=?"];
        NSString *encMsg = [cleanMsg stringByAddingPercentEncodingWithAllowedCharacters:allowed] ?: @"";

        NSString *urlStr = (digits.length > 0)
            ? [NSString stringWithFormat:@"whatsapp://send?phone=%@&text=%@", digits, encMsg]
            : [NSString stringWithFormat:@"whatsapp://send?text=%@", encMsg];
        NSURL *waURL = [NSURL URLWithString:urlStr];
        NSURL *fallbackURL = (digits.length > 0)
            ? [NSURL URLWithString:[NSString stringWithFormat:@"https://wa.me/%@?text=%@", digits, encMsg]]
            : [NSURL URLWithString:[NSString stringWithFormat:@"https://wa.me/?text=%@", encMsg]];

        dispatch_async(dispatch_get_main_queue(), ^{
            [[UIApplication sharedApplication] openURL:waURL options:@{} completionHandler:^(BOOL ok) {
                if (ok) {
                    if (completion) completion(YES, display, (digits.length > 0), waURL);
                } else if (fallbackURL != nil) {
                    [[UIApplication sharedApplication] openURL:fallbackURL options:@{} completionHandler:^(BOOL ok2) {
                        if (completion) completion(ok2, display, (digits.length > 0), fallbackURL);
                    }];
                } else {
                    if (completion) completion(NO, display, (digits.length > 0), waURL);
                }
            }];
        });
    });
}

static inline NSString *kronosTryGenerateChatViaLocalMac14B(NSArray<NSDictionary *> *messagesPayload,
                                                            NSUInteger maxTokens,
                                                            float temperature) {
    KronosMobileSettings *s = [KronosMobileSettings sharedSettings];
    if (!s.macSyncEnabled || !s.preferMacEngineOnLan || s.macServerURL.length == 0) return nil;
    NSString *base = [s.macServerURL stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@"/ "]];

    // Fast 0.65s LAN reachability probe so away-from-Wi-Fi fallback to on-device Metal is instantaneous.
    NSURL *probeURL = [NSURL URLWithString:[base stringByAppendingString:@"/v1/device/status"]];
    if (probeURL != nil) {
        NSMutableURLRequest *probeReq = [NSMutableURLRequest requestWithURL:probeURL
                                                                cachePolicy:NSURLRequestReloadIgnoringLocalCacheData
                                                            timeoutInterval:0.65];
        dispatch_semaphore_t pSem = dispatch_semaphore_create(0);
        __block BOOL reachable = NO;
        [[[NSURLSession sharedSession] dataTaskWithRequest:probeReq completionHandler:^(NSData *d, NSURLResponse *r, NSError *e) {
            (void)d;
            if (e == nil && [r isKindOfClass:[NSHTTPURLResponse class]] && [(NSHTTPURLResponse *)r statusCode] == 200) {
                reachable = YES;
            }
            dispatch_semaphore_signal(pSem);
        }] resume];
        dispatch_semaphore_wait(pSem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.75 * NSEC_PER_SEC)));
        if (!reachable) return nil;
    }

    NSURL *url = [NSURL URLWithString:[base stringByAppendingString:@"/v1/chat/completions"]];
    if (url == nil) return nil;

    NSDictionary *bodyObj = @{
        @"messages": messagesPayload ?: @[],
        @"max_tokens": @(maxTokens),
        @"temperature": @(temperature),
    };
    NSData *bodyData = [NSJSONSerialization dataWithJSONObject:bodyObj options:0 error:nil];
    if (bodyData == nil) return nil;

    NSMutableURLRequest *req = [NSMutableURLRequest requestWithURL:url
                                                       cachePolicy:NSURLRequestReloadIgnoringLocalCacheData
                                                   timeoutInterval:7.5];
    req.HTTPMethod = @"POST";
    [req setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
    req.HTTPBody = bodyData;

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
    dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(7.8 * NSEC_PER_SEC)));

    if (statusCode == 200 && respData.length > 0) {
        NSDictionary *json = [NSJSONSerialization JSONObjectWithData:respData options:0 error:nil];
        NSArray *choices = [json isKindOfClass:[NSDictionary class]] ? json[@"choices"] : nil;
        if ([choices isKindOfClass:[NSArray class]] && choices.count > 0) {
            NSDictionary *first = choices.firstObject;
            NSDictionary *msg = [first isKindOfClass:[NSDictionary class]] ? first[@"message"] : nil;
            NSString *content = [msg isKindOfClass:[NSDictionary class]] ? msg[@"content"] : nil;
            if ([content isKindOfClass:[NSString class]] && content.length > 0) {
                [KronosIosDeviceContextManager sharedManager].macReachable = YES;
                return content;
            }
        }
    }
    return nil;
}

static inline BOOL kronosParseLinkedInPostIntent(NSString *userText,
                                                 NSString **outTopic,
                                                 BOOL *outWantsPhoto) {
    if (kronosIsInterrogativeOrLookupText(userText)) return NO;
    NSString *trimmed = [kronosNormalizeApostrophes(userText ?: @"") stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSString *low = [trimmed lowercaseString];

    BOOL mentionsLinkedIn = [low containsString:@"linkedin"] || [low containsString:@"linkendin"] ||
                            [low containsString:@"linked in"] || [low containsString:@"linkedln"];
    if (!mentionsLinkedIn) return NO;

    BOOL isPostAction = [low containsString:@"post"] || [low containsString:@"create"] ||
                        [low containsString:@"write"] || [low containsString:@"draft"] ||
                        [low containsString:@"share"] || [low containsString:@"make"] ||
                        [low containsString:@"publish"];
    if (!isPostAction || [low hasPrefix:@"open linkedin"] || [low hasPrefix:@"open my linkedin"] ||
        [low hasPrefix:@"what is my linkedin"] || [low hasPrefix:@"whats my linkedin"]) {
        return NO;
    }

    BOOL wantsPhoto = [low containsString:@"picture"] || [low containsString:@"photo"] ||
                      [low containsString:@"image"] || [low containsString:@"pic"] ||
                      [low containsString:@"shot"] || [low containsString:@"snap"] ||
                      [low containsString:@"camera"];

    NSString *topic = @"Monday motivation and software engineering focus";
    NSArray<NSString *> *topicMarkers = @[@"talk about ", @"talking about ", @"about ", @"on the topic of ", @"regarding "];
    for (NSString *mk in topicMarkers) {
        NSRange mr = [low rangeOfString:mk];
        if (mr.location != NSNotFound) {
            NSString *rawTopic = [trimmed substringFromIndex:mr.location + mr.length];
            NSString *rawLow = [rawTopic lowercaseString];
            NSArray<NSString *> *cutoffs = @[@" should also", @" and ask ", @" then ask ", @". should "];
            for (NSString *co in cutoffs) {
                NSRange cr = [rawLow rangeOfString:co];
                if (cr.location != NSNotFound) {
                    rawTopic = [rawTopic substringToIndex:cr.location];
                    rawLow = [rawTopic lowercaseString];
                }
            }
            rawTopic = kronosCleanExtractedSpan(rawTopic);
            if (rawTopic.length > 0) {
                topic = rawTopic;
                break;
            }
        }
    }

    if (outTopic) *outTopic = topic;
    if (outWantsPhoto) *outWantsPhoto = wantsPhoto;
    return YES;
}

// iOS Device Action Router (Open WhatsApp, GitHub, LinkedIn, Safari, Mail, Maps, Notes, etc.)
typedef void (^KronosDeviceActionCompletion)(BOOL handled, NSString *actionSummary, NSString *spokenReply);

static inline void kronosTryExecuteIosDeviceAction(NSString *userText, KronosDeviceActionCompletion completion) {
    NSString *trimmed = [userText stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSString *lower = [trimmed lowercaseString];

    // Defer multi-step WhatsApp drafting and LinkedIn post creation to their dedicated workflows.
    if (kronosParseWhatsAppDraftIntent(trimmed, NULL, NULL, NULL, NULL, NULL) ||
        kronosParseLinkedInPostIntent(trimmed, NULL, NULL)) {
        completion(NO, nil, nil);
        return;
    }

    BOOL isOpenCmd = [lower hasPrefix:@"open "] || [lower hasPrefix:@"launch "] ||
                     [lower hasPrefix:@"go to "] || [lower hasPrefix:@"take me to "] ||
                     [lower hasPrefix:@"start "];
    BOOL isSearchCmd = [lower hasPrefix:@"search google for "] || [lower hasPrefix:@"search the web for "] ||
                       [lower hasPrefix:@"search safari for "] || [lower hasPrefix:@"google "];

    if (!isOpenCmd && !isSearchCmd) {
        completion(NO, nil, nil);
        return;
    }

    if (![KronosMobileSettings sharedSettings].allowPhoneUse) {
        completion(YES,
                   @"Phone Use Disabled",
                   @"Allow Phone Use is currently turned off in Kronos Settings. Tap the gear icon in the top bar or sidebar to enable Phone Use.");
        return;
    }

    if (isSearchCmd) {
        NSString *q = trimmed;
        NSArray<NSString *> *prefixes = @[@"search google for ", @"search the web for ", @"search safari for ", @"google "];
        for (NSString *p in prefixes) {
            if ([lower hasPrefix:p]) {
                q = [trimmed substringFromIndex:p.length];
                break;
            }
        }
        NSString *enc = [q stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLQueryAllowedCharacterSet]] ?: @"";
        NSURL *url = [NSURL URLWithString:[NSString stringWithFormat:@"https://www.google.com/search?q=%@", enc]];
        [[UIApplication sharedApplication] openURL:url options:@{} completionHandler:^(BOOL ok) {
            (void)ok;
            completion(YES,
                       [NSString stringWithFormat:@"Searched Safari for \"%@\"", q],
                       [NSString stringWithFormat:@"Opening Safari search results for %@.", q]);
        }];
        return;
    }

    NSDictionary<NSString *, NSArray<NSString *> *> *appMap = @{
        @"whatsapp":  @[@"WhatsApp",  @"whatsapp://app",              @"https://wa.me"],
        @"whats app": @[@"WhatsApp",  @"whatsapp://app",              @"https://wa.me"],
        @"telegram":  @[@"Telegram",  @"tg://",                       @"https://web.telegram.org"],
        @"slack":     @[@"Slack",     @"slack://open",                @"https://slack.com"],
        @"youtube":   @[@"YouTube",   @"youtube://",                  @"https://m.youtube.com"],
        @"instagram": @[@"Instagram", @"instagram://app",             @"https://www.instagram.com"],
        @"twitter":   @[@"X",         @"twitter://",                  @"https://x.com/Melody_Leonard7"],
        @"x app":     @[@"X",         @"twitter://",                  @"https://x.com/Melody_Leonard7"],
        @"github":    @[@"GitHub",    @"https://github.com/melodyLeonard", @"https://github.com/melodyLeonard"],
        @"linkedin":  @[@"LinkedIn",  @"https://www.linkedin.com/in/melody-leonard-a362b6192", @"https://www.linkedin.com/in/melody-leonard-a362b6192"],
        @"mail":      @[@"Mail",      @"mailto:",                     @"mailto:melodyleonard7@gmail.com"],
        @"email":     @[@"Mail",      @"mailto:",                     @"mailto:melodyleonard7@gmail.com"],
        @"messages":  @[@"Messages",  @"sms:",                        @"sms:"],
        @"imessage":  @[@"Messages",  @"sms:",                        @"sms:"],
        @"phone":     @[@"Phone",     @"tel:",                        @"tel:"],
        @"maps":      @[@"Maps",      @"maps://",                     @"https://maps.apple.com"],
        @"notes":     @[@"Notes",     @"mobilenotes://",              @"mobilenotes://"],
        @"reminders": @[@"Reminders", @"x-apple-reminderkit://",      @"x-apple-reminderkit://"],
        @"calendar":  @[@"Calendar",  @"calshow://",                  @"calshow://"],
        @"photos":    @[@"Photos",    @"photos-redirect://",          @"photos-redirect://"],
        @"files":     @[@"Files",     @"shareddocuments://",          @"shareddocuments://"],
        @"shortcuts": @[@"Shortcuts", @"shortcuts://",                @"shortcuts://"],
        @"safari":    @[@"Safari",    @"https://www.google.com",      @"https://www.google.com"],
        @"settings":  @[@"Settings",  UIApplicationOpenSettingsURLString, UIApplicationOpenSettingsURLString],
    };

    for (NSString *key in appMap.allKeys) {
        if ([lower containsString:key]) {
            NSArray<NSString *> *spec = appMap[key];
            NSString *label = spec[0];
            NSURL *primaryURL = [NSURL URLWithString:spec[1]];
            NSURL *fallbackURL = [NSURL URLWithString:spec[2]];
            [[UIApplication sharedApplication] openURL:primaryURL options:@{} completionHandler:^(BOOL ok) {
                if (ok) {
                    completion(YES,
                               [NSString stringWithFormat:@"Opened %@", label],
                               [NSString stringWithFormat:@"Opened %@.", label]);
                } else if (fallbackURL != nil && ![fallbackURL isEqual:primaryURL]) {
                    [[UIApplication sharedApplication] openURL:fallbackURL options:@{} completionHandler:^(BOOL ok2) {
                        (void)ok2;
                        completion(YES,
                                   [NSString stringWithFormat:@"Opened %@ in browser", label],
                                   [NSString stringWithFormat:@"%@ app is not installed on this iPhone, so I opened %@ in Safari.", label, label]);
                    }];
                } else {
                    completion(YES,
                               [NSString stringWithFormat:@"%@ unavailable", label],
                               [NSString stringWithFormat:@"I tried to open %@, which is not currently installed on this iPhone.", label]);
                }
            }];
            return;
        }
    }

    // Direct URL opening (e.g. "open https://..." or "open github.com")
    NSArray<NSString *> *words = [trimmed componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
    NSString *lastWord = words.lastObject ?: @"";
    if ([lastWord hasPrefix:@"http://"] || [lastWord hasPrefix:@"https://"] || [lastWord containsString:@".com"] || [lastWord containsString:@".org"] || [lastWord containsString:@".io"]) {
        NSString *urlStr = ([lastWord hasPrefix:@"http://"] || [lastWord hasPrefix:@"https://"])
            ? lastWord
            : [@"https://" stringByAppendingString:lastWord];
        NSURL *url = [NSURL URLWithString:urlStr];
        if (url != nil) {
            [[UIApplication sharedApplication] openURL:url options:@{} completionHandler:^(BOOL ok) {
                (void)ok;
                completion(YES,
                           [NSString stringWithFormat:@"Opened %@", lastWord],
                           [NSString stringWithFormat:@"Opened %@.", lastWord]);
            }];
            return;
        }
    }

    completion(NO, nil, nil);
}

static inline BOOL kronosIsAgenticWorkspacePrompt(NSString *userText) {
    NSString *lower = [userText lowercaseString];
    if (kronosParseWhatsAppDraftIntent(userText, NULL, NULL, NULL, NULL, NULL) ||
        kronosParseLinkedInPostIntent(userText, NULL, NULL)) {
        return NO;
    }
    NSArray<NSString *> *triggers = @[
        @"create a file", @"write a file", @"save to ", @"save as ", @"save this to ",
        @"create index.html", @"build a webpage", @"create a webpage", @"create a website",
        @"build a website", @"build a web app", @"create a portfolio", @"make a webpage",
        @"make a website", @"edit file", @"update index.html", @"create a markdown",
        @"write a script file", @"save summary"
    ];
    for (NSString *t in triggers) {
        if ([lower containsString:t]) return YES;
    }
    NSArray<NSString *> *exts = @[@".html", @".css", @".js", @".md", @".txt", @".json", @".py", @".rs"];
    BOOL mentionsExt = NO;
    for (NSString *e in exts) {
        if ([lower containsString:e]) { mentionsExt = YES; break; }
    }
    if (mentionsExt && ([lower containsString:@"create "] || [lower containsString:@"write "] ||
                        [lower containsString:@"edit "] || [lower containsString:@"save "] ||
                        [lower containsString:@"update "] || [lower containsString:@"build "])) {
        return YES;
    }
    return NO;
}

static inline NSString *kronosExtractFallbackFileIfNeeded(NSString *userText, NSString *modelReply) {
    NSArray<NSString *> *parts = [modelReply componentsSeparatedByString:@"```"];
    if (parts.count < 3) return nil;
    NSString *block = parts[1];
    NSString *lang = @"txt";
    NSRange nl = [block rangeOfString:@"\n"];
    if (nl.location != NSNotFound && nl.location < 18) {
        NSString *first = [[block substringToIndex:nl.location] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
        if (first.length > 0) lang = [first lowercaseString];
        block = [block substringFromIndex:nl.location + 1];
    }
    if (block.length < 16) return nil;

    NSString *targetName = nil;
    NSArray<NSString *> *tokens = [userText componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    for (NSString *tok in tokens) {
        NSString *clean = [tok stringByTrimmingCharactersInSet:[NSCharacterSet punctuationCharacterSet]];
        if ([clean containsString:@"."] && clean.length >= 4 && clean.length <= 48 && ![clean hasPrefix:@"http"]) {
            targetName = clean.lastPathComponent;
            break;
        }
    }
    if (targetName == nil) {
        if ([lang isEqualToString:@"html"] || [block containsString:@"<!DOCTYPE"] || [block containsString:@"<html"]) {
            targetName = @"index.html";
        } else if ([lang isEqualToString:@"css"]) {
            targetName = @"styles.css";
        } else if ([lang isEqualToString:@"js"] || [lang isEqualToString:@"javascript"]) {
            targetName = @"app.js";
        } else if ([lang isEqualToString:@"python"] || [lang isEqualToString:@"py"]) {
            targetName = @"script.py";
        } else if ([lang isEqualToString:@"rust"] || [lang isEqualToString:@"rs"]) {
            targetName = @"main.rs";
        } else if ([lang isEqualToString:@"json"]) {
            targetName = @"data.json";
        } else {
            targetName = @"notes.md";
        }
    }
    NSString *full = [kronosWorkspaceDirectory() stringByAppendingPathComponent:targetName];
    if (![[NSFileManager defaultManager] fileExistsAtPath:full]) {
        [block writeToFile:full atomically:YES encoding:NSUTF8StringEncoding error:nil];
    }
    return targetName;
}

// On-Device Live Voice Input (SFSpeechRecognizer + AVAudioEngine) and Spoken Replies (AVSpeechSynthesizer)
@protocol KronosVoiceControllerDelegate <NSObject>
- (void)voiceControllerDidChangeListening:(BOOL)isListening statusText:(NSString *)status;
- (void)voiceControllerDidTranscribePartial:(NSString *)transcript;
- (void)voiceControllerDidFinishUtterance:(NSString *)finalTranscript;
@end

@interface KronosVoiceController : NSObject
@property (nonatomic, weak) id<KronosVoiceControllerDelegate> delegate;
@property (nonatomic, strong) SFSpeechRecognizer *speechRecognizer;
@property (nonatomic, strong) AVAudioEngine *audioEngine;
@property (nonatomic, strong) SFSpeechAudioBufferRecognitionRequest *recognitionRequest;
@property (nonatomic, strong) SFSpeechRecognitionTask *recognitionTask;
@property (nonatomic, strong) AVSpeechSynthesizer *synthesizer;
@property (nonatomic, strong) NSTimer *silenceTimer;
@property (nonatomic, copy) NSString *latestTranscript;
@property (nonatomic, assign) BOOL isListening;
@property (nonatomic, assign) BOOL voiceOutputEnabled;
- (void)toggleListening;
- (void)stopListeningAndCommit:(BOOL)shouldCommit;
- (void)speakText:(NSString *)text;
- (void)stopSpeaking;
@end

@implementation KronosVoiceController
- (instancetype)init {
    self = [super init];
    if (self) {
        _speechRecognizer = [[SFSpeechRecognizer alloc] initWithLocale:[NSLocale localeWithLocaleIdentifier:@"en-US"]]
            ?: [[SFSpeechRecognizer alloc] init];
        _audioEngine = [[AVAudioEngine alloc] init];
        _synthesizer = [[AVSpeechSynthesizer alloc] init];
        _voiceOutputEnabled = [KronosMobileSettings sharedSettings].spokenRepliesEnabled;
    }
    return self;
}

- (void)stopSpeaking {
    if (self.synthesizer.isSpeaking) {
        [self.synthesizer stopSpeakingAtBoundary:AVSpeechBoundaryImmediate];
    }
}

- (void)speakText:(NSString *)text {
    if (text.length == 0) return;
    [self stopSpeaking];

    NSArray<NSString *> *parts = [text componentsSeparatedByString:@"```"];
    NSMutableString *clean = [NSMutableString string];
    for (NSUInteger i = 0; i < parts.count; i++) {
        if (i % 2 == 0) {
            [clean appendString:parts[i]];
        } else {
            [clean appendString:@" Code snippet shown on screen. "];
        }
    }
    NSString *spoken = [clean stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (spoken.length == 0) return;

    AVAudioSession *session = [AVAudioSession sharedInstance];
    [session setCategory:AVAudioSessionCategoryPlayback
             withOptions:AVAudioSessionCategoryOptionDuckOthers
                   error:nil];
    [session setActive:YES error:nil];

    AVSpeechUtterance *utt = [AVSpeechUtterance speechUtteranceWithString:spoken];
    utt.voice = [AVSpeechSynthesisVoice voiceWithLanguage:@"en-US"];
    utt.rate = AVSpeechUtteranceDefaultSpeechRate;
    [self.synthesizer speakUtterance:utt];
}

- (void)toggleListening {
    if (self.isListening) {
        [self stopListeningAndCommit:[KronosMobileSettings sharedSettings].autoSendVoiceAfterPause];
        return;
    }
    [self stopSpeaking];

    [SFSpeechRecognizer requestAuthorization:^(SFSpeechRecognizerAuthorizationStatus authStatus) {
        dispatch_async(dispatch_get_main_queue(), ^{
            if (authStatus != SFSpeechRecognizerAuthorizationStatusAuthorized) {
                [self.delegate voiceControllerDidChangeListening:NO statusText:@"Enable Speech Recognition in Settings"];
                return;
            }
            [[AVAudioSession sharedInstance] requestRecordPermission:^(BOOL granted) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    if (!granted) {
                        [self.delegate voiceControllerDidChangeListening:NO statusText:@"Enable Microphone access in Settings"];
                        return;
                    }
                    [self startAudioCaptureSession];
                });
            }];
        });
    }];
}

- (void)resetSilenceTimer {
    [self.silenceTimer invalidate];
    if (![KronosMobileSettings sharedSettings].autoSendVoiceAfterPause) {
        return;
    }
    __weak typeof(self) weakSelf = self;
    self.silenceTimer = [NSTimer scheduledTimerWithTimeInterval:1.65 repeats:NO block:^(NSTimer *timer) {
        (void)timer;
        __strong typeof(weakSelf) strongSelf = weakSelf;
        if (strongSelf != nil && strongSelf.isListening && strongSelf.latestTranscript.length > 0) {
            [strongSelf stopListeningAndCommit:YES];
        }
    }];
}

- (void)startAudioCaptureSession {
    [self stopListeningAndCommit:NO];
    self.latestTranscript = @"";

    NSError *err = nil;
    AVAudioSession *audioSession = [AVAudioSession sharedInstance];
    [audioSession setCategory:AVAudioSessionCategoryPlayAndRecord
                         mode:AVAudioSessionModeMeasurement
                      options:(AVAudioSessionCategoryOptionDefaultToSpeaker | AVAudioSessionCategoryOptionAllowBluetoothHFP)
                        error:&err];
    [audioSession setActive:YES withOptions:AVAudioSessionSetActiveOptionNotifyOthersOnDeactivation error:&err];

    self.recognitionRequest = [[SFSpeechAudioBufferRecognitionRequest alloc] init];
    self.recognitionRequest.shouldReportPartialResults = YES;

    AVAudioInputNode *inputNode = self.audioEngine.inputNode;
    [inputNode removeTapOnBus:0];
    AVAudioFormat *recordingFormat = [inputNode outputFormatForBus:0];
    if (recordingFormat.sampleRate <= 0 || recordingFormat.channelCount == 0) {
        [self.delegate voiceControllerDidChangeListening:NO statusText:@"Microphone unavailable"];
        return;
    }

    __weak typeof(self) weakSelf = self;
    [inputNode installTapOnBus:0 bufferSize:1024 format:recordingFormat block:^(AVAudioPCMBuffer *buffer, AVAudioTime *when) {
        (void)when;
        [weakSelf.recognitionRequest appendAudioPCMBuffer:buffer];
    }];

    [self.audioEngine prepare];
    if (![self.audioEngine startAndReturnError:&err]) {
        [inputNode removeTapOnBus:0];
        [self.delegate voiceControllerDidChangeListening:NO statusText:@"Could not start microphone"];
        return;
    }

    self.isListening = YES;
    [self.delegate voiceControllerDidChangeListening:YES statusText:@"Listening..."];

    self.recognitionTask = [self.speechRecognizer recognitionTaskWithRequest:self.recognitionRequest
                                                               resultHandler:^(SFSpeechRecognitionResult *result, NSError *error) {
        dispatch_async(dispatch_get_main_queue(), ^{
            __strong typeof(weakSelf) strongSelf = weakSelf;
            if (strongSelf == nil || !strongSelf.isListening) return;
            if (result != nil) {
                NSString *text = result.bestTranscription.formattedString ?: @"";
                strongSelf.latestTranscript = text;
                [strongSelf.delegate voiceControllerDidTranscribePartial:text];
                if (text.length > 0) {
                    [strongSelf resetSilenceTimer];
                }
                if (result.isFinal) {
                    [strongSelf stopListeningAndCommit:[KronosMobileSettings sharedSettings].autoSendVoiceAfterPause];
                    return;
                }
            }
            if (error != nil) {
                BOOL autoCommit = [KronosMobileSettings sharedSettings].autoSendVoiceAfterPause && (strongSelf.latestTranscript.length > 0);
                [strongSelf stopListeningAndCommit:autoCommit];
            }
        });
    }];
}

- (void)stopListeningAndCommit:(BOOL)shouldCommit {
    [self.silenceTimer invalidate];
    self.silenceTimer = nil;

    BOOL wasListening = self.isListening;
    self.isListening = NO;

    if (self.audioEngine.isRunning) {
        [self.audioEngine stop];
        [self.audioEngine.inputNode removeTapOnBus:0];
    }
    [self.recognitionRequest endAudio];
    [self.recognitionTask cancel];
    self.recognitionRequest = nil;
    self.recognitionTask = nil;

    if (wasListening) {
        [self.delegate voiceControllerDidChangeListening:NO statusText:nil];
        NSString *finalText = [self.latestTranscript stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        self.latestTranscript = @"";
        if (shouldCommit && finalText.length > 0) {
            [self.delegate voiceControllerDidFinishUtterance:finalText];
        }
    }
}
@end

// Background bi-directional continual memory and 14B Teacher distillation vault sync over local Wi-Fi.
static inline void kronosSyncContinualMemoryAndDistillVaultWithMac(void) {
    KronosMobileSettings *s = [KronosMobileSettings sharedSettings];
    if (!s.macSyncEnabled || s.macServerURL.length == 0) return;
    NSString *base = [s.macServerURL stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@"/ "]];
    if (base.length == 0) return;

    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        // 1. Push local memory store & merge Mac's response (last-write-wins per fact key + union of notes/reminders).
        NSURL *memURL = [NSURL URLWithString:[base stringByAppendingString:@"/v1/device/memory"]];
        if (memURL != nil) {
            NSDictionary *localMem = kronosLoadMemoryStore();
            NSData *body = [NSJSONSerialization dataWithJSONObject:localMem options:0 error:nil];
            if (body.length > 0) {
                NSMutableURLRequest *req = [NSMutableURLRequest requestWithURL:memURL
                                                                   cachePolicy:NSURLRequestReloadIgnoringLocalCacheData
                                                               timeoutInterval:2.0];
                req.HTTPMethod = @"POST";
                [req setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
                req.HTTPBody = body;
                [[[NSURLSession sharedSession] dataTaskWithRequest:req completionHandler:^(NSData *data, NSURLResponse *resp, NSError *err) {
                    if (err == nil && [resp isKindOfClass:[NSHTTPURLResponse class]] && [(NSHTTPURLResponse *)resp statusCode] == 200 && data.length > 0) {
                        id merged = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
                        if ([merged isKindOfClass:[NSDictionary class]]) {
                            kronosMergeMemoryStoreFromRemoteDict((NSDictionary *)merged);
                        }
                    }
                }] resume];
            }
        }

        // 2. Push local 14B Teacher distillation vault & merge curated Mac Teacher demonstrations.
        NSURL *vaultURL = [NSURL URLWithString:[base stringByAppendingString:@"/v1/device/distill_vault"]];
        if (vaultURL != nil) {
            NSArray *localVault = kronosLoadDistilledTeacherVault();
            NSData *vBody = [NSJSONSerialization dataWithJSONObject:localVault options:0 error:nil];
            if (vBody.length > 0) {
                NSMutableURLRequest *vReq = [NSMutableURLRequest requestWithURL:vaultURL
                                                                    cachePolicy:NSURLRequestReloadIgnoringLocalCacheData
                                                                timeoutInterval:2.5];
                vReq.HTTPMethod = @"POST";
                [vReq setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
                vReq.HTTPBody = vBody;
                [[[NSURLSession sharedSession] dataTaskWithRequest:vReq completionHandler:^(NSData *data, NSURLResponse *resp, NSError *err) {
                    if (err == nil && [resp isKindOfClass:[NSHTTPURLResponse class]] && [(NSHTTPURLResponse *)resp statusCode] == 200 && data.length > 0) {
                        id merged = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
                        if ([merged isKindOfClass:[NSArray class]]) {
                            kronosMergeDistilledVaultFromRemoteArray((NSArray *)merged);
                        }
                    }
                }] resume];
            }
        }
    });
}

// Conversational Follow-Up, Ellipsis ("How about Oliver-ke", "And Sandra?"), and Pronoun Anaphora Resolver.
static inline NSString *kronosExtractPrimaryPersonFromResolvedTurn(NSString *turnQuery) {
    if (turnQuery.length == 0) return nil;
    if (kronosIsExternalMessageLookupQuery(turnQuery)) {
        NSArray<NSString *> *terms = kronosExtractExternalLookupTerms(turnQuery);
        if (terms.firstObject.length > 0) {
            return kronosFormatContactDisplayName(terms.firstObject);
        }
    }
    NSString *contactTarget = nil;
    if (kronosIsContactPhoneOrEmailQuery(turnQuery, &contactTarget) && contactTarget.length > 0) {
        return contactTarget;
    }
    NSString *waRecip = nil;
    if (kronosParseWhatsAppDraftIntent(turnQuery, &waRecip, NULL, NULL, NULL, NULL) && waRecip.length > 0) {
        return waRecip;
    }
    return nil;
}

static inline NSString *kronosReplaceFollowUpPronouns(NSString *trimmed, NSString *entity) {
    if (trimmed.length == 0 || entity.length == 0) return trimmed;
    NSSet<NSString *> *possessedNouns = [NSSet setWithArray:@[
        @"phone", @"telephone", @"mobile", @"cell", @"number", @"contact", @"digits",
        @"email", @"address", @"message", @"messages", @"last", @"latest", @"recent",
        @"chat", @"whatsapp", @"imessage", @"telegram", @"birthday", @"age", @"office", @"handle"
    ]];
    NSArray<NSString *> *spaceParts = [trimmed componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
    NSMutableArray<NSString *> *words = [NSMutableArray arrayWithCapacity:spaceParts.count];
    for (NSString *sp in spaceParts) {
        if (sp.length > 0) [words addObject:sp];
    }
    BOOL replaced = NO;
    for (NSUInteger i = 0; i < words.count; i++) {
        NSString *w = words[i];
        NSString *trailPunct = @"";
        while (w.length > 1) {
            unichar lastCh = [w characterAtIndex:(w.length - 1)];
            if (lastCh == '?' || lastCh == '.' || lastCh == ',') {
                trailPunct = [[NSString stringWithFormat:@"%C", lastCh] stringByAppendingString:trailPunct];
                w = [w substringToIndex:(w.length - 1)];
            } else {
                break;
            }
        }
        NSString *wl = [w lowercaseString];
        if ([wl isEqualToString:@"his"] || [wl isEqualToString:@"their"]) {
            words[i] = [NSString stringWithFormat:@"%@'s%@", entity, trailPunct];
            replaced = YES;
        } else if ([wl isEqualToString:@"him"] || [wl isEqualToString:@"them"]) {
            words[i] = [NSString stringWithFormat:@"%@%@", entity, trailPunct];
            replaced = YES;
        } else if ([wl isEqualToString:@"her"]) {
            NSString *nextClean = @"";
            if (i + 1 < words.count) {
                nextClean = [[words[i + 1] lowercaseString] stringByTrimmingCharactersInSet:[NSCharacterSet punctuationCharacterSet]];
            }
            if (trailPunct.length == 0 && [possessedNouns containsObject:nextClean]) {
                words[i] = [NSString stringWithFormat:@"%@'s", entity];
            } else {
                words[i] = [NSString stringWithFormat:@"%@%@", entity, trailPunct];
            }
            replaced = YES;
        }
    }
    return replaced ? [words componentsJoinedByString:@" "] : trimmed;
}

static inline NSString *kronosResolveFollowUpUserQuery(NSString *rawUserText, NSArray<NSDictionary *> *convMsgs) {
    NSString *trimmed = [kronosNormalizeApostrophes(rawUserText ?: @"")
                         stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (trimmed.length == 0 || convMsgs.count == 0) return rawUserText;
    if (kronosTrySaveUserTaughtFact(trimmed, NULL)) return rawUserText;

    NSDictionary *prevUserDict = nil;
    for (NSDictionary *m in [convMsgs reverseObjectEnumerator]) {
        if ([m isKindOfClass:[NSDictionary class]] && [m[@"role"] isEqualToString:@"user"]) {
            prevUserDict = m;
            break;
        }
    }
    if (prevUserDict == nil) return rawUserText;
    NSString *prevQuery = prevUserDict[@"resolvedQuery"];
    if (![prevQuery isKindOfClass:[NSString class]] || prevQuery.length == 0) {
        prevQuery = prevUserDict[@"displayText"] ?: prevUserDict[@"content"];
    }
    if (![prevQuery isKindOfClass:[NSString class]] || prevQuery.length == 0) return rawUserText;

    NSString *low = [trimmed lowercaseString];
    NSString *prevEntity = kronosExtractPrimaryPersonFromResolvedTurn(prevQuery);

    // 1. Pronoun anaphora resolution ("What is her phone number?", "Message him on WhatsApp").
    if (prevEntity.length > 0 && trimmed.length <= 120) {
        NSArray<NSString *> *alphaWords = kronosSplitAlphaWords(low);
        BOOL hasPronoun = [alphaWords containsObject:@"him"] || [alphaWords containsObject:@"her"] ||
                          [alphaWords containsObject:@"his"] || [alphaWords containsObject:@"them"] ||
                          [alphaWords containsObject:@"their"];
        if (hasPronoun) {
            NSString *candidate = kronosReplaceFollowUpPronouns(trimmed, prevEntity);
            if (![candidate isEqualToString:trimmed]) {
                if (kronosIsContactPhoneOrEmailQuery(candidate, NULL) ||
                    kronosIsExternalMessageLookupQuery(candidate) ||
                    kronosParseWhatsAppDraftIntent(candidate, NULL, NULL, NULL, NULL, NULL) ||
                    kronosLookupVerifiedLocalFact(candidate) != nil) {
                    return candidate;
                }
            }
        }
    }

    // 2. Ellipsis follow-up resolution ("How about Oliver-ke", "What about Sandra?", "And Dad?", "Check Oliver-ke").
    if (trimmed.length > 72) return rawUserText;
    if (kronosIsExternalMessageLookupQuery(trimmed) ||
        kronosIsContactPhoneOrEmailQuery(trimmed, NULL) ||
        kronosParseWhatsAppDraftIntent(trimmed, NULL, NULL, NULL, NULL, NULL) ||
        kronosParseLinkedInPostIntent(trimmed, NULL, NULL)) {
        return rawUserText;
    }

    NSArray<NSString *> *ellipsisPrefixes = @[
        @"and what about for ", @"and how about for ", @"and what about ", @"and how about ",
        @"how about for ", @"what about for ", @"how about from ", @"what about from ",
        @"how about ", @"what about ", @"what of ", @"how of ",
        @"and from ", @"and for ", @"also check ", @"and ", @"try ", @"check "
    ];
    NSString *rawTarget = nil;
    for (NSString *ep in ellipsisPrefixes) {
        if ([low hasPrefix:ep]) {
            if ([ep isEqualToString:@"and "] && trimmed.length > 36 && ![trimmed hasSuffix:@"?"]) {
                continue;
            }
            rawTarget = [trimmed substringFromIndex:ep.length];
            break;
        }
    }

    // Bare contact follow-up ("Oliver-ke?", "Sandra?") immediately after a message or contact lookup.
    if (rawTarget == nil && [trimmed hasSuffix:@"?"] && trimmed.length <= 28 &&
        (kronosIsExternalMessageLookupQuery(prevQuery) || kronosIsContactPhoneOrEmailQuery(prevQuery, NULL))) {
        NSArray<NSString *> *wsParts = [trimmed componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
        if (wsParts.count >= 1 && wsParts.count <= 2) {
            NSSet<NSString *> *nonEntityFirstWords = [NSSet setWithArray:@[
                @"why", @"how", @"when", @"where", @"what", @"whats", @"who", @"which",
                @"yes", @"no", @"ok", @"okay", @"really", @"sure", @"thanks", @"hello", @"hi"
            ]];
            NSString *firstAlpha = kronosSplitAlphaWords(wsParts.firstObject).firstObject ?: @"";
            if (firstAlpha.length >= 2 && ![nonEntityFirstWords containsObject:firstAlpha]) {
                rawTarget = trimmed;
            }
        }
    }

    if (rawTarget == nil) return rawUserText;

    NSString *prevLow = [kronosNormalizeApostrophes(prevQuery) lowercaseString];
    NSString *channel = @"WhatsApp";
    if ([low containsString:@"imessage"] || [low containsString:@" sms"]) {
        channel = @"iMessage";
    } else if ([low containsString:@"telegram"]) {
        channel = @"Telegram";
    } else if ([low containsString:@"slack"]) {
        channel = @"Slack";
    } else if ([prevLow containsString:@"imessage"]) {
        channel = @"iMessage";
    } else if ([prevLow containsString:@"telegram"]) {
        channel = @"Telegram";
    } else if ([prevLow containsString:@"slack"]) {
        channel = @"Slack";
    }

    NSCharacterSet *trimPunct = [NSCharacterSet characterSetWithCharactersInString:@" \t\n\r?.,\"'"];
    rawTarget = [rawTarget stringByTrimmingCharactersInSet:trimPunct];
    NSArray<NSString *> *trailFillers = @[
        @" on whatsapp", @" on whats app", @" on imessage", @" on telegram", @" on slack",
        @" too", @" as well", @" instead", @" please", @"'s"
    ];
    BOOL strippedFiller = YES;
    while (strippedFiller && rawTarget.length > 0) {
        strippedFiller = NO;
        NSString *rtLow = [rawTarget lowercaseString];
        for (NSString *tf in trailFillers) {
            if ([rtLow hasSuffix:tf] && rawTarget.length > tf.length + 1) {
                rawTarget = [[rawTarget substringToIndex:(rawTarget.length - tf.length)] stringByTrimmingCharactersInSet:trimPunct];
                strippedFiller = YES;
                break;
            }
        }
    }

    if (rawTarget.length < 2 || rawTarget.length > 42) return rawUserText;
    NSArray<NSString *> *targetSpaceWords = [rawTarget componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
    if (targetSpaceWords.count > 5) return rawUserText;

    NSString *paddedTargetLow = [NSString stringWithFormat:@" %@ ", [rawTarget lowercaseString]];
    NSArray<NSString *> *clauseVerbs = @[
        @" is ", @" was ", @" were ", @" are ", @" did ", @" does ", @" can ", @" could ",
        @" should ", @" would ", @" will ", @" have ", @" has ", @" write ", @" create ", @" build ", @" explain "
    ];
    for (NSString *cv in clauseVerbs) {
        if ([paddedTargetLow containsString:cv]) return rawUserText;
    }

    if (kronosIsExternalMessageLookupQuery(prevQuery)) {
        return [NSString stringWithFormat:@"What was %@'s last message to me on %@?", rawTarget, channel];
    }

    NSString *prevContact = nil;
    if (kronosIsContactPhoneOrEmailQuery(prevQuery, &prevContact)) {
        NSString *tLow = [rawTarget lowercaseString];
        if (([tLow isEqualToString:@"email"] || [tLow isEqualToString:@"email address"] || [tLow hasSuffix:@" email"]) && prevContact.length > 0) {
            return [NSString stringWithFormat:@"What is %@'s email?", prevContact];
        }
        if (([tLow isEqualToString:@"phone"] || [tLow isEqualToString:@"number"] || [tLow isEqualToString:@"phone number"]) && prevContact.length > 0) {
            return [NSString stringWithFormat:@"What is %@'s phone number?", prevContact];
        }
        NSString *attr = [prevLow containsString:@"email"] ? @"email" : @"phone number";
        return [NSString stringWithFormat:@"What is %@'s %@?", rawTarget, attr];
    }

    BOOL prevWasFactRecall = kronosIsUnverifiedPersonalFactQuery(prevQuery, NULL) || (kronosLookupVerifiedLocalFact(prevQuery) != nil);
    if (prevWasFactRecall) {
        NSString *tLow = [rawTarget lowercaseString];
        if ([tLow hasPrefix:@"my "] || [tLow containsString:@"'s "]) {
            return [NSString stringWithFormat:@"What is %@?", rawTarget];
        }
        if (kronosLookupVerifiedLocalFact(rawTarget) != nil) {
            return [NSString stringWithFormat:@"What is my %@?", rawTarget];
        }
    }

    return rawUserText;
}

#endif
