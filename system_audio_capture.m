#import <Foundation/Foundation.h>
#import <ScreenCaptureKit/ScreenCaptureKit.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreAudio/CoreAudioTypes.h>
#import <objc/runtime.h>
#include <unistd.h>

@interface AudioOutput : NSObject <SCStreamOutput>
@end

@implementation AudioOutput
- (void)stream:(SCStream *)stream
        didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
        ofType:(SCStreamOutputType)type {
    if (type != SCStreamOutputTypeAudio || !CMSampleBufferIsValid(sampleBuffer)) return;

    size_t size = 0;
    CMBlockBufferRef blockBuffer = NULL;
    CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sampleBuffer, &size, NULL, 0, kCFAllocatorDefault, kCFAllocatorDefault,
        kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment, &blockBuffer);

    AudioBufferList *list = malloc(size);
    OSStatus status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sampleBuffer, NULL, list, size, kCFAllocatorDefault, kCFAllocatorDefault,
        kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment, &blockBuffer);
    if (status == noErr) {
        for (UInt32 i = 0; i < list->mNumberBuffers; i++) {
            AudioBuffer buffer = list->mBuffers[i];
            if (buffer.mData && buffer.mDataByteSize) {
                write(STDOUT_FILENO, buffer.mData, buffer.mDataByteSize);
            }
        }
    }
    if (blockBuffer) CFRelease(blockBuffer);
    free(list);
}
@end

int main(void) {
    @autoreleasepool {
        [SCShareableContent
            getShareableContentExcludingDesktopWindows:NO
            onScreenWindowsOnly:NO
            completionHandler:^(SCShareableContent *content, NSError *error) {
                if (error || content.displays.count == 0) {
                    fprintf(stderr, "ERROR: %s\n",
                            (error.localizedDescription ?: @"No display available").UTF8String);
                    exit(1);
                }

                SCContentFilter *filter = [[SCContentFilter alloc]
                    initWithDisplay:content.displays.firstObject
                    excludingApplications:@[] exceptingWindows:@[]];
                SCStreamConfiguration *config = [SCStreamConfiguration new];
                config.capturesAudio = YES;
                config.excludesCurrentProcessAudio = YES;
                config.sampleRate = 16000;
                config.channelCount = 1;
                config.width = 2;
                config.height = 2;
                config.minimumFrameInterval = CMTimeMake(1, 1);

                AudioOutput *receiver = [AudioOutput new];
                SCStream *stream = [[SCStream alloc] initWithFilter:filter
                    configuration:config delegate:nil];
                NSError *addError = nil;
                [stream addStreamOutput:receiver type:SCStreamOutputTypeAudio
                    sampleHandlerQueue:dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0)
                    error:&addError];
                if (addError) {
                    fprintf(stderr, "ERROR: %s\n", addError.localizedDescription.UTF8String);
                    exit(1);
                }
                [stream startCaptureWithCompletionHandler:^(NSError *startError) {
                    if (startError) {
                        fprintf(stderr, "ERROR: %s\n", startError.localizedDescription.UTF8String);
                        exit(1);
                    }
                    fprintf(stderr, "READY\n");
                }];

                // Keep strong references alive for the lifetime of the process.
                objc_setAssociatedObject(NSProcessInfo.processInfo, @"stream", stream,
                                         OBJC_ASSOCIATION_RETAIN_NONATOMIC);
                objc_setAssociatedObject(NSProcessInfo.processInfo, @"receiver", receiver,
                                         OBJC_ASSOCIATION_RETAIN_NONATOMIC);
            }];
        dispatch_main();
    }
}
